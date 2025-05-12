import os
from dotenv import load_dotenv
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel, field_validator
import asyncio
from typing import List, Dict, Optional, Any, Union
import json
from openai import OpenAI
import tiktoken
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import fitz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
def create_db_engine() -> Engine:
    DATABASE_URL = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(DATABASE_URL)

engine = create_db_engine()

# Create database schema - drop and recreate table
def setup_database():
    with engine.connect() as conn:
        try:
            # Drop the existing table if it exists
            conn.execute(text("DROP TABLE IF EXISTS contracts;"))
            conn.commit()
            
            # Create extension if not exists
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create the table with all required columns
            conn.execute(text("""
                CREATE TABLE contracts (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX contracts_embedding_idx 
                ON contracts USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            conn.commit()
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise

# Initialize database
setup_database()

# Define output types
class DocumentAnalysis(BaseModel):
    has_legal_content: bool
    document_type: str
    key_topics: List[str]
    reasoning: str

class SearchResult(BaseModel):
    relevant_chunks: Union[List[Dict], Dict]
    confidence_score: float
    reasoning: str
    
    @field_validator('relevant_chunks')
    @classmethod
    def validate_relevant_chunks(cls, v):
        # If we get a dict, convert it to a list with a single item
        if isinstance(v, dict):
            return [v]
        return v

class ContractAnswer(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    suggested_next_steps: Optional[List[str]]

# Define the agents
document_analyzer_agent = Agent(
    name="Document Analyzer",
    handoff_description="Analyzes document content and determines its type and key topics",
    instructions="""Analyze the document content to:
    1. Determine if it contains legal content
    2. Identify the document type (contract, agreement, policy, etc.)
    3. Extract key topics and themes
    4. Provide reasoning for your analysis""",
    output_type=DocumentAnalysis
)

semantic_search_agent = Agent(
    name="Semantic Search",
    handoff_description="Performs semantic search to find relevant contract sections",
    instructions="""Search for relevant contract sections by:
    1. Understanding the query intent
    2. Finding semantically similar content
    3. Ranking results by relevance
    4. Providing confidence scores
    
    IMPORTANT: Your output must include 'relevant_chunks' as a list of dictionaries, not a single dictionary.
    For example:
    {
      "relevant_chunks": [
        {"content": "first chunk", "similarity": 0.9},
        {"content": "second chunk", "similarity": 0.8}
      ],
      "confidence_score": 0.85,
      "reasoning": "These sections contain information about..."
    }""",
    output_type=SearchResult
)

contract_analyst_agent = Agent(
    name="Contract Analyst",
    handoff_description="Provides detailed analysis of contract sections",
    instructions="""Analyze contract sections to:
    1. Provide clear, accurate answers
    2. Cite specific sections and clauses
    3. Maintain professional tone
    4. Show empathy for user concerns
    5. Never hallucinate information
    6. Suggest next steps when needed""",
    output_type=ContractAnswer
)

# Define guardrails
async def document_guardrail(ctx, agent, input_data):
    result = await Runner.run(document_analyzer_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(DocumentAnalysis)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.has_legal_content,
    )

# Main RAG agent with handoffs
rag_agent = Agent(
    name="RAG System",
    instructions="""You are a contract analysis system that:
    1. Analyzes documents for legal content
    2. Performs semantic search for relevant sections
    3. Provides detailed, accurate answers
    4. Maintains professional tone
    5. Cites specific sections
    6. Shows empathy for user concerns""",
    handoffs=[document_analyzer_agent, semantic_search_agent, contract_analyst_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=document_guardrail),
    ]
)

# Helper functions for agents
class PDFProcessor:
    def __init__(self, max_file_size_mb: int = 10):
        self.max_file_size = max_file_size_mb * 1024 * 1024
    
    def process_document(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            if len(content) > self.max_file_size:
                raise ValueError(f"File size exceeds {self.max_file_size/1024/1024}MB limit")
            
            doc = fitz.open(stream=content, filetype="pdf")
            text = []
            for page in doc:
                text.append(page.get_text())
            return {
                "title": os.path.basename(file_path),
                "content": "\n".join(text),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "error": str(e)}

class EmbeddingManager:
    def __init__(self, client: OpenAI, engine: Engine):
        self.client = client
        self.engine = engine
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[Dict]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_text = self.tokenizer.decode(tokens[i : i + max_tokens])
            chunks.append({
                "text": chunk_text,
                "start_token": i,
                "end_token": min(i + max_tokens, len(tokens))
            })
        return chunks
    
    def store_chunks(self, title: str, chunks: List[Dict]) -> Dict:
        stored_chunks = 0
        for chunk in chunks:
            try:
                embedding_result = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk["text"]
                )
                embedding = embedding_result.data[0].embedding
                
                with self.engine.connect() as conn:
                    try:
                        # Try to store with vector type
                        conn.execute(
                            text("""
                                INSERT INTO contracts (title, content, embedding, metadata)
                                VALUES (:title, :content, :embedding, :metadata)
                            """),
                            {
                                "title": title,
                                "content": chunk["text"],
                                "embedding": embedding,
                                "metadata": json.dumps({
                                    "start_token": chunk["start_token"],
                                    "end_token": chunk["end_token"]
                                })
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Vector insert failed, trying JSON: {e}")
                        # Fallback to JSON if vector type fails
                        conn.execute(
                            text("""
                                INSERT INTO contracts (title, content, embedding, metadata)
                                VALUES (:title, :content, :embedding, :metadata)
                            """),
                            {
                                "title": title,
                                "content": chunk["text"],
                                "embedding": json.dumps(embedding),
                                "metadata": json.dumps({
                                    "start_token": chunk["start_token"],
                                    "end_token": chunk["end_token"]
                                })
                            }
                        )
                    conn.commit()
                stored_chunks += 1
            except Exception as e:
                logger.error(f"Error storing chunk: {e}")
                continue
        return {"status": "success", "chunks_stored": stored_chunks}

    def search_similar_chunks(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for chunks semantically similar to the query"""
        try:
            # Generate embedding for the query
            query_embedding_result = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = query_embedding_result.data[0].embedding
            
            # Fallback to regular SQL search if vector operations aren't working
            with self.engine.connect() as conn:
                # First, check if we have vector extension support
                try:
                    # Try using vector operations if available
                    result = conn.execute(
                        text("""
                            SELECT id, title, content, metadata,
                            1 - (embedding <=> :query_embedding) as similarity
                            FROM contracts
                            ORDER BY similarity DESC
                            LIMIT :limit
                        """),
                        {"query_embedding": query_embedding, "limit": limit}
                    )
                except Exception as e:
                    logger.warning(f"Vector operations not supported: {e}")
                    # Fallback to regular SQL search
                    result = conn.execute(
                        text("""
                            SELECT id, title, content, metadata,
                            1 - (embedding <=> :query_embedding) as similarity
                            FROM contracts
                            ORDER BY similarity DESC
                            LIMIT :limit
                        """),
                        {"query_embedding": query_embedding, "limit": limit}
                    )
            
            # Process the result
            similar_chunks = []
            for row in result:
                similar_chunks.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "metadata": row[3],
                    "similarity": row[4]
                })
            
            return similar_chunks
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

# Initialize helpers
pdf_processor = PDFProcessor()
embedding_manager = EmbeddingManager(client, engine)

# Simplified process_contract function to use only agents
async def process_contract(pdf_file: str, question: str) -> Dict:
    try:
        # Process document
        print("Processing document...")
        doc_result = pdf_processor.process_document(pdf_file)
        if doc_result["status"] == "error":
            raise Exception(f"Failed to process document: {doc_result['error']}")
        
        # Generate embeddings and store chunks
        print("Generating embeddings...")
        chunks = embedding_manager.chunk_text(doc_result["content"])
        store_result = embedding_manager.store_chunks(doc_result["title"], chunks)
        print(f"Stored {store_result['chunks_stored']} chunks in the database")
        
        # Find relevant chunks for the question
        print("Finding relevant chunks...")
        try:
            relevant_chunks = embedding_manager.search_similar_chunks(question)
            relevant_texts = [chunk["content"] for chunk in relevant_chunks]
            
            # If no chunks found, use first few chunks as context
            if not relevant_texts:
                relevant_texts = [chunk["text"] for chunk in chunks[:3]]
        except Exception as e:
            logger.warning(f"Error during semantic search: {e}")
            # Fallback to using first chunks
            relevant_texts = [chunk["text"] for chunk in chunks[:3]]
        
        # Prepare input for the RAG agent - convert to a list of messages instead of a dict
        input_messages = [
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocument: {doc_result['title']}\n\nContent: {doc_result['content'][:5000]}\n\nRelevant chunks: {json.dumps(relevant_texts)}"
            }
        ]
        
        # Run the main RAG agent with handoffs
        print("Running RAG agent with handoffs...")
        try:
            result = await Runner.run(rag_agent, input_messages)
            
            # Add debug logging
            print(f"Agent result type: {type(result)}")
            print(f"Agent result keys: {dir(result)}")
            
            # Since RunResult doesn't have handoff_history, we'll extract information from available attributes
            last_agent = result.last_agent
            print(f"Last agent: {last_agent.name if last_agent else 'None'}")
            
            # Extract the final output
            final_output = result.final_output
            print(f"Final output type: {type(final_output)}")
            
            # Try to parse the final output based on the last agent
            if last_agent and last_agent.name == "Contract Analyst":
                try:
                    # Try to get typed output using final_output_as
                    answer_output = result.final_output_as(ContractAnswer)
                    
                    return {
                        "answer": answer_output.answer,
                        "citations": answer_output.citations,
                        "confidence": answer_output.confidence,
                        "agent": last_agent.name
                    }
                except Exception as e:
                    print(f"Error parsing final output: {e}")
                    # If parsing fails, return raw output
                    return {
                        "answer": str(final_output),
                        "citations": [],
                        "confidence": 0.5,
                        "agent": last_agent.name if last_agent else "Unknown"
                    }
            else:
                # Return the raw final output if the last agent wasn't Contract Analyst
                return {
                    "answer": str(final_output),
                    "citations": [],
                    "confidence": 0.5,
                    "agent": last_agent.name if last_agent else "RAG System"
                }
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error during agent execution: {e}")
            print(f"Traceback: {traceback_str}")
            raise
    
    except Exception as e:
        logger.error(f"Error in contract processing: {e}")
        raise

# Example usage
async def main():
    print("Contract Analysis System")
    print("=======================")
    
    pdf_file = input("Enter the path to your PDF file: ")
    question = input("Enter your question about the contract: ")
    
    try:
        print("\nProcessing your request...")
        result = await process_contract(pdf_file, question)
        
        print("\nAnswer:", result["answer"])
        
        if result.get("citations"):
            print("\nCitations:", ", ".join(result["citations"]))
        
        if result.get("confidence"):
            print(f"\nConfidence: {result['confidence']:.2f}")
        
        if result.get("agent"):
            print(f"\nAgent: {result['agent']}")
        
        # Allow follow-up questions
        while True:
            print("\n" + "-"*50)
            follow_up = input("\nAsk a follow-up question (or type 'exit' to quit): ")
            if follow_up.lower() == 'exit':
                break
            
            result = await process_contract(pdf_file, follow_up)
            
            print("\nAnswer:", result["answer"])
            if result.get("citations"):
                print("\nCitations:", ", ".join(result["citations"]))
            if result.get("confidence"):
                print(f"\nConfidence: {result['confidence']:.2f}")
            if result.get("agent"):
                print(f"\nAgent: {result['agent']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
    