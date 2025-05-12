import os
from dotenv import load_dotenv
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel
import asyncio
from typing import List, Dict, Optional
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
    relevant_chunks: List[Dict]
    confidence_score: float
    reasoning: str

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
    4. Providing confidence scores""",
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
                embedding = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk["text"]
                ).data[0].embedding
                
                with self.engine.connect() as conn:
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
                    conn.commit()
                stored_chunks += 1
            except Exception as e:
                logger.error(f"Error storing chunk: {e}")
                continue
        return {"status": "success", "chunks_stored": stored_chunks}

# Initialize helpers
pdf_processor = PDFProcessor()
embedding_manager = EmbeddingManager(client, engine)

# Simplified process_contract function to avoid agent issues
async def process_contract(pdf_file: str, question: str) -> str:
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
        
        # For now, bypass the agent and use direct OpenAI query
        print("Analyzing document and generating response...")
        
        # Extract relevant chunks for the question
        relevant_texts = []
        for chunk in chunks:
            if question.lower() in chunk["text"].lower():
                relevant_texts.append(chunk["text"])
        
        # If no direct matches, use all chunks
        if not relevant_texts:
            relevant_texts = [chunk["text"] for chunk in chunks[:3]]
        
        # Generate response using OpenAI directly
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a contract analysis assistant. Answer questions about contracts clearly and accurately. 
                    Cite specific sections when possible. If payment amounts are mentioned, specify the exact amounts."""
                },
                {
                    "role": "user", 
                    "content": f"Contract content:\n\n{' '.join(relevant_texts)}\n\nQuestion: {question}"
                }
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error in contract processing: {e}")
        raise

# Example usage
async def main():
    pdf_file = input("Enter the path to your PDF file: ")
    question = input("Enter your question about the contract: ")
    
    try:
        answer = await process_contract(pdf_file, question)
        print("\nAssistant:", answer)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 