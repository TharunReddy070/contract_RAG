import os
import json
import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union

import fitz  # PyMuPDF
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, Field
from openai import AsyncOpenAI # Use AsyncOpenAI for async operations
from openai.types import FunctionDefinition # Assuming this is what 'agents' uses for tool defs

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Assuming 'agents' module and its components are defined elsewhere and work as expected
# from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
# For this example, let's create dummy classes for Agent and Runner to make the script runnable
class Agent:
    def __init__(self, name: str, instructions: str, output_type: BaseModel, tools: List[Any]):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.tools = tools
        logger.info(f"Agent '{name}' initialized with tools: {[tool.__name__ for tool in tools]}")

class Runner:
    @staticmethod
    async def run(agent: Agent, inputs: Dict[str, Any]) -> 'RunResult': # Use a placeholder RunResult
        logger.info(f"Running agent '{agent.name}' with inputs: {list(inputs.keys())}")
        # This is a mock. In reality, it would call the LLM with agent instructions, inputs, and tools.
        # The LLM would then decide to call a tool or return a direct answer.
        # For simplicity, we'll assume it tries to use the first tool if inputs match,
        # or directly constructs the output_type for demonstration.

        # Mocking tool execution
        if agent.name == "Contract Type Analyzer" and "document" in inputs:
            tool_result = extract_contract_type(text=inputs["document"])
            return RunResult(agent.output_type(**tool_result))
        elif agent.name == "Field Extractor" and "document" in inputs and "contract_type" in inputs:
            tool_result = extract_contract_fields(text=inputs["document"], contract_type=inputs["contract_type"])
            return RunResult(agent.output_type(**tool_result))
        elif agent.name == "RAG Query Handler" and "question" in inputs:
            # Mock RAG by just creating a dummy answer
            return RunResult(ContractAnswer(answer=f"Mock answer to '{inputs['question']}'", citations=[], confidence=0.75))
        
        # Fallback if no specific logic matches
        try:
            # Attempt to create a dummy instance of the output_type
            # This requires output_type to have default values or be simple enough
            if agent.output_type == ContractType:
                mock_output = ContractType(contract_type="UNKNOWN", confidence=0.1, reasoning="Mock fallback")
            elif agent.output_type == ContractFields:
                # Provide all required fields with dummy data
                mock_output = ContractFields(
                    contract_type="UNKNOWN", client_name="N/A", seller_name="N/A",
                    client_address="N/A", seller_address="N/A", property_address="N/A",
                    commencement_date="1970-01-01", expiration_date="1970-01-01",
                    governing_law="N/A", notice_period="N/A", late_payment_penalty="N/A"
                )
            elif agent.output_type == ContractAnswer:
                mock_output = ContractAnswer(answer="No specific agent logic matched.", citations=[], confidence=0.2)
            else:
                raise NotImplementedError(f"Mocking for {agent.output_type} not implemented")
            return RunResult(mock_output)
        except Exception as e:
            logger.error(f"Error mocking agent {agent.name} output: {e}")
            # Depending on output_type, this might fail if it has required fields without defaults
            return RunResult(agent.output_type())


class RunResult: # Placeholder for what Runner.run might return
    def __init__(self, output: Any):
        self._output = output

    def final_output_as(self, output_type: BaseModel) -> BaseModel:
        if isinstance(self._output, output_type):
            return self._output
        # In a real scenario, this might involve parsing/validation from a raw LLM response
        raise TypeError(f"Output is not of type {output_type}")

    @property
    def final_output(self) -> Any: # General access to output
        return self._output


# --- Configuration ---
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_USER = os.getenv("DB_USER", "user")
DB_PASS = os.getenv("DB_PASS", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "contractsdb")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 # For text-embedding-3-small

# --- Logging Setup ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY not set.")

# --- OpenAI Client ---
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Database Setup ---
db_engine = create_async_engine(DATABASE_URL, echo=LOG_LEVEL == "DEBUG")
AsyncSessionFactory = sessionmaker(
    bind=db_engine, class_=AsyncSession, expire_on_commit=False
)

async def initialize_schema():
    """Initializes the database schema if tables/extensions don't exist."""
    async with db_engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS contracts (
                    id SERIAL PRIMARY KEY,
                    contract_type TEXT,
                    title TEXT,
                    content TEXT,
                    embedding vector({EMBEDDING_DIMENSION}),
                    
                    client_name TEXT,
                    seller_name TEXT,
                    client_address TEXT,
                    seller_address TEXT,
                    property_address TEXT,
                    
                    commencement_date DATE,
                    expiration_date DATE,
                    
                    purchase_price DECIMAL(12,2),
                    monthly_rent DECIMAL(10,2),
                    security_deposit DECIMAL(10,2),
                    
                    governing_law TEXT,
                    notice_period TEXT,
                    late_payment_penalty TEXT,
                    
                    metadata JSONB,
                    confidence_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            # Consider creating indexes only if they don't exist or use a migration tool
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS contracts_embedding_idx 
                ON contracts USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS contracts_type_idx ON contracts(contract_type);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS contracts_client_idx ON contracts(client_name);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS contracts_dates_idx ON contracts(commencement_date, expiration_date);"))
            
            logger.info("Database schema initialized successfully (if not already present).")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

# --- Pydantic Models for API/Data Structure ---
class ContractType(BaseModel):
    contract_type: str = Field(..., description="Type of the contract (e.g., LEASE, RENT, BUY/SELL)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the classification")
    reasoning: str = Field(..., description="Reasoning behind the classification")

class ContractFields(BaseModel):
    contract_type: str # This comes from the previous step
    client_name: str
    seller_name: str
    client_address: str
    seller_address: str
    property_address: str
    commencement_date: date # Use date type
    expiration_date: date   # Use date type
    purchase_price: Optional[float] = None
    monthly_rent: Optional[float] = None
    security_deposit: Optional[float] = None
    governing_law: str
    notice_period: str
    late_payment_penalty: str

    @field_validator('commencement_date', 'expiration_date', mode='before')
    @classmethod
    def parse_date(cls, value: Any) -> date:
        if isinstance(value, str):
            try:
                return datetime.strptime(value, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        if isinstance(value, date):
            return value
        raise TypeError("Invalid type for date field")

class ContractAnswer(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


# --- Custom Tools (Functions for Agents) ---
@FunctionDefinition
async def extract_contract_type(text: str) -> Dict[str, Any]:
    """
    Analyzes contract text to determine if it's a lease, rent, or buy/sell contract.
    Returns the contract type, confidence score, and reasoning.
    
    NOTE: This is a STUB. In production, this would involve an LLM call.
    Example:
    response = await aclient.chat.completions.create(
        model="gpt-3.5-turbo", # Or your preferred model
        messages=[
            {"role": "system", "content": "You are an expert contract analyzer. Determine if the contract is a LEASE, RENT, or BUY/SELL agreement. Provide confidence and reasoning."},
            {"role": "user", "content": f"Contract text: {text[:4000]}"} # Truncate for context window
        ],
        # Potentially use function calling here for structured output
    )
    # Parse response to fit ContractType model
    """
    logger.info("Tool 'extract_contract_type' called (using MOCK implementation).")
    # Mocked implementation
    if "lease agreement" in text.lower():
        return {"contract_type": "LEASE", "confidence": 0.95, "reasoning": "Contains 'lease agreement'"}
    elif "rental agreement" in text.lower():
        return {"contract_type": "RENT", "confidence": 0.92, "reasoning": "Contains 'rental agreement'"}
    elif "purchase agreement" in text.lower() or "sale agreement" in text.lower():
        return {"contract_type": "BUY/SELL", "confidence": 0.98, "reasoning": "Contains 'purchase agreement' or 'sale agreement'"}
    return {"contract_type": "UNKNOWN", "confidence": 0.5, "reasoning": "Keywords not found in mock."}


@FunctionDefinition
async def extract_contract_fields(text: str, contract_type: str) -> Dict[str, Any]:
    """
    Extracts specific fields from the contract based on its type.
    Returns a dictionary of extracted fields.
    
    NOTE: This is a STUB. In production, this would involve an LLM call,
    possibly tailored by contract_type.
    """
    logger.info(f"Tool 'extract_contract_fields' called for type '{contract_type}' (using MOCK implementation).")
    # Mocked implementation
    return {
        "client_name": "John Doe (Mock)",
        "seller_name": "Jane Smith (Mock)",
        "client_address": "123 Main St, Anytown (Mock)",
        "seller_address": "456 Oak Ave, Anytown (Mock)",
        "property_address": "789 Pine Rd, Anytown (Mock)",
        "commencement_date": "2024-01-01",
        "expiration_date": "2025-01-01",
        "purchase_price": 500000.00 if contract_type == "BUY/SELL" else None,
        "monthly_rent": 2000.00 if contract_type in ["LEASE", "RENT"] else None,
        "security_deposit": 4000.00 if contract_type in ["LEASE", "RENT"] else None,
        "governing_law": "State of Mockifornia",
        "notice_period": "30 days (Mock)",
        "late_payment_penalty": "5% of monthly rent (Mock)"
    }

@FunctionDefinition
async def search_contract_chunks(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches for relevant chunks in the contract database using vector similarity.
    Returns a list of matching chunks with similarity scores.
    """
    logger.info(f"Tool 'search_contract_chunks' called with query: '{query}', k={k}")
    try:
        embedding_response = await aclient.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding for query '{query}': {e}")
        return []

    async with AsyncSessionFactory() as session:
        try:
            stmt = text("""
                SELECT content, metadata, contract_type, title,
                1 - (embedding <=> :query_embedding) as similarity
                FROM contracts
                ORDER BY similarity DESC
                LIMIT :k
            """)
            result = await session.execute(stmt, {"query_embedding": query_embedding, "k": k})
            rows = result.fetchall()
            
            return [{
                "content": row.content,
                "metadata": row.metadata,
                "contract_type": row.contract_type,
                "title": row.title,
                "similarity": row.similarity
            } for row in rows]
        except Exception as e:
            logger.error(f"Error searching contract chunks in DB: {e}")
            return []


# --- Helper Functions ---
async def _get_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        # fitz operations are synchronous, run in a thread pool
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        text_parts = []
        for page_num in range(len(doc)):
            page = await asyncio.to_thread(doc.load_page, page_num)
            text_parts.append(await asyncio.to_thread(page.get_text))
        full_text = "\n".join(text_parts)
        logger.info(f"Successfully extracted text from {pdf_path} (length: {len(full_text)} chars).")
        return full_text
    except Exception as e:
        logger.error(f"Error processing PDF file {pdf_path}: {e}")
        raise


async def _save_contract_data_to_db(
    contract_type_info: ContractType,
    contract_fields: ContractFields,
    full_text: str,
    pdf_title: str
) -> int:
    """Stores extracted contract data and its embedding in the database."""
    logger.info(f"Preparing to save contract data for: {pdf_title}")
    try:
        embedding_response = await aclient.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=full_text # Embed the full text, or consider chunking for very large docs
        )
        embedding = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding for contract content: {e}")
        raise

    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                stmt = text("""
                    INSERT INTO contracts (
                        contract_type, title, content, embedding,
                        client_name, seller_name, client_address, seller_address,
                        property_address, commencement_date, expiration_date,
                        purchase_price, monthly_rent, security_deposit,
                        governing_law, notice_period, late_payment_penalty,
                        metadata, confidence_score, updated_at
                    ) VALUES (
                        :contract_type, :title, :content, :embedding,
                        :client_name, :seller_name, :client_address, :seller_address,
                        :property_address, :commencement_date, :expiration_date,
                        :purchase_price, :monthly_rent, :security_deposit,
                        :governing_law, :notice_period, :late_payment_penalty,
                        :metadata, :confidence_score, :updated_at
                    ) RETURNING id
                """)
                
                db_data = {
                    "contract_type": contract_type_info.contract_type,
                    "title": pdf_title,
                    "content": full_text, # Storing full text, can be large
                    "embedding": embedding,
                    "client_name": contract_fields.client_name,
                    "seller_name": contract_fields.seller_name,
                    "client_address": contract_fields.client_address,
                    "seller_address": contract_fields.seller_address,
                    "property_address": contract_fields.property_address,
                    "commencement_date": contract_fields.commencement_date,
                    "expiration_date": contract_fields.expiration_date,
                    "purchase_price": contract_fields.purchase_price,
                    "monthly_rent": contract_fields.monthly_rent,
                    "security_deposit": contract_fields.security_deposit,
                    "governing_law": contract_fields.governing_law,
                    "notice_period": contract_fields.notice_period,
                    "late_payment_penalty": contract_fields.late_payment_penalty,
                    "metadata": json.dumps({
                        "extraction_timestamp": datetime.now().isoformat(),
                        "source_file": pdf_title,
                        "contract_type_reasoning": contract_type_info.reasoning
                    }),
                    "confidence_score": contract_type_info.confidence,
                    "updated_at": datetime.now()
                }
                
                result = await session.execute(stmt, db_data)
                contract_id = result.scalar_one()
                await session.commit()
                logger.info(f"Contract data for '{pdf_title}' stored successfully with ID: {contract_id}.")
                return contract_id
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving contract data to DB: {e}")
                raise

# --- Agent Definitions ---
# (Ensure Agent and Runner are properly imported or defined)
# Assuming 'tools' argument in Agent takes a list of function objects decorated with @FunctionDefinition
contract_type_agent = Agent(
    name="Contract Type Analyzer",
    instructions="Analyze the document to determine if it's a lease, rent, or buy/sell contract. Use the extract_contract_type tool.",
    output_type=ContractType, # Pydantic model for expected output structure
    tools=[extract_contract_type] # List of available tools for this agent
)

field_extractor_agent = Agent(
    name="Field Extractor",
    instructions="Extract specific fields from the contract based on its type. Use the extract_contract_fields tool.",
    output_type=ContractFields,
    tools=[extract_contract_fields]
)

rag_query_agent = Agent(
    name="RAG Query Handler",
    instructions="Answer questions about contracts using semantic search over stored contract data. Use the search_contract_chunks tool to find relevant information, then synthesize an answer.",
    output_type=ContractAnswer,
    tools=[search_contract_chunks]
)

# --- Main Processing Logic ---
async def process_contract_and_query(pdf_file_path: str, question: str) -> Optional[ContractAnswer]:
    """
    Processes a contract PDF: extracts text, classifies, extracts fields, stores, and answers a question.
    """
    try:
        contract_text = await _get_text_from_pdf(pdf_file_path)
    except Exception as e:
        logger.error(f"Failed to read or process PDF {pdf_file_path}: {e}")
        return None

    # 1. Analyze Contract Type
    logger.info("Running Contract Type Analyzer agent...")
    type_result_obj = await Runner.run(
        contract_type_agent,
        {"document": contract_text}
    )
    contract_type_info = type_result_obj.final_output_as(ContractType)
    logger.info(f"Contract type identified: {contract_type_info.contract_type} (Confidence: {contract_type_info.confidence})")

    # 2. Extract Fields
    logger.info("Running Field Extractor agent...")
    fields_result_obj = await Runner.run(
        field_extractor_agent,
        {
            "document": contract_text,
            "contract_type": contract_type_info.contract_type
        }
    )
    # The output type for field_extractor_agent is ContractFields, which requires contract_type.
    # The tool `extract_contract_fields` needs to return it, or we need to inject it.
    # For now, let's assume the tool's output is merged or it doesn't need contract_type internally.
    # The Pydantic model ContractFields requires contract_type. Let's add it from the prior step.
    extracted_fields_data = fields_result_obj.final_output_as(ContractFields)
    # Ensure contract_type from previous step is part of the final fields object if not already set by agent
    if not hasattr(extracted_fields_data, 'contract_type') or not extracted_fields_data.contract_type:
         extracted_fields_data.contract_type = contract_type_info.contract_type
    
    logger.info(f"Extracted fields for '{contract_type_info.contract_type}' contract.")

    # 3. Store Contract Data
    pdf_title = os.path.basename(pdf_file_path)
    try:
        contract_id = await _save_contract_data_to_db(
            contract_type_info=contract_type_info,
            contract_fields=extracted_fields_data,
            full_text=contract_text,
            pdf_title=pdf_title
        )
        if contract_id:
            logger.info(f"Contract '{pdf_title}' processed and stored with ID: {contract_id}")
        else:
            logger.warning(f"Contract '{pdf_title}' processed but failed to store.")
            # Decide if to proceed with query if storage fails
    except Exception as e:
        logger.error(f"Failed to store contract data for {pdf_title}: {e}")
        # Decide if to proceed: For RAG, data needs to be in DB.
        # If the query is only about the *current* document, maybe it can proceed without DB.
        # However, search_contract_chunks queries the DB.
        return None 

    # 4. Handle User Query using RAG
    if not question:
        logger.info("No question provided. Skipping RAG query.")
        return ContractAnswer(answer="Contract processed and stored. No question asked.", citations=[], confidence=1.0)

    logger.info(f"Running RAG Query Handler agent for question: '{question}'")
    # The RAG agent might need access to the just-processed data, or it might rely solely on DB search.
    # Current search_contract_chunks searches the whole DB.
    rag_input = {
        "question": question,
        # Optionally, pass context if agent can use it before DB search:
        # "current_contract_text": contract_text,
        # "current_contract_fields": extracted_fields_data.model_dump() 
    }
    answer_result_obj = await Runner.run(rag_query_agent, rag_input)
    final_answer = answer_result_obj.final_output_as(ContractAnswer)
    
    logger.info(f"RAG Answer: {final_answer.answer} (Confidence: {final_answer.confidence})")
    return final_answer


# --- Example Usage ---
async def main():
    # Initialize schema once at startup
    await initialize_schema()

    # For testing, create a dummy PDF if one doesn't exist
    dummy_pdf_path = "dummy_lease_contract.pdf"
    if not os.path.exists(dummy_pdf_path):
        try:
            doc = fitz.open() # new empty PDF
            page = doc.new_page()
            page.insert_text((72, 72), "This is a DUMMY Lease Agreement for testing purposes.\nClient: John Doe\nLandlord: Jane Smith PropCo")
            page.insert_text((72, 144), "The monthly rent is $2000. Commencement date: 2024-01-01. Expiration date: 2025-01-01.")
            doc.save(dummy_pdf_path)
            logger.info(f"Created dummy PDF: {dummy_pdf_path}")
        except Exception as e:
            logger.error(f"Could not create dummy PDF: {e}")
            return

    pdf_file = input(f"Enter contract PDF path (default: {dummy_pdf_path}): ") or dummy_pdf_path
    question = input("Enter your question (e.g., 'What is the monthly rent?'): ")

    if not os.path.exists(pdf_file):
        logger.error(f"The PDF file '{pdf_file}' does not exist. Exiting.")
        return

    try:
        result = await process_contract_and_query(pdf_file, question)
        if result:
            print("\n--- Query Result ---")
            print(f"Answer: {result.answer}")
            if result.citations:
                print(f"Citations: {result.citations}")
            print(f"Confidence: {result.confidence:.2f}")
        else:
            print("\nFailed to process the contract or answer the question. Check logs for details.")

    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shut down by user.")
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)