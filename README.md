# Contract Manager - Agentic RAG System

A sophisticated contract analysis system leveraging Retrieval-Augmented Generation (RAG) with OpenAI's Agents SDK to provide intelligent contract analysis.

## Features

- **Document Processing**: Automatic processing of PDF contracts
- **Semantic Search**: Finding the most relevant contract sections for any query
- **Multi-Agent Architecture**: Specialized agents for different aspects of contract analysis
- **Database Integration**: PostgreSQL with vector storage for semantic search
- **Intelligent Analysis**: Detailed answers with citations and confidence scores

## Architecture

The system uses a multi-agent architecture with three specialized agents:

1. **Document Analyzer Agent**: Analyzes document content to determine document type and key topics
2. **Semantic Search Agent**: Performs semantic search to find the most relevant contract sections
3. **Contract Analyst Agent**: Provides detailed analysis and answers based on the retrieved content

All agents work together in a coordinated workflow managed by the OpenAI Agents SDK.

## Requirements

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key
- Required Python packages (see requirements.txt)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

Ensure PostgreSQL is installed with the pgvector extension. Then set the following environment variables:

```
DB_USER=your_db_user
DB_PASS=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
OPENAI_API_KEY=your_openai_api_key
```

You can use a `.env` file in the project directory for these settings.

### 3. Database Configuration

The system will automatically configure the database schema when it first runs.

## Usage

Run the application:

```bash
python further.py
```

The application will prompt you to:
1. Enter the path to a PDF contract
2. Enter your question about the contract

After processing, the system will provide:
- A detailed answer to your question
- Citations to specific contract sections
- A confidence score for the answer
- The name of the agent that provided the answer

You can then ask follow-up questions.

## How It Works

1. **Document Processing**: The PDF is processed and chunked into manageable sections
2. **Embedding Generation**: Each chunk is converted to an embedding and stored in the database
3. **Query Processing**: When you ask a question, it is processed by the agent system:
   - Document Analyzer examines the content type and structure
   - Semantic Search finds the most relevant contract sections
   - Contract Analyst generates the final answer with citations

## Troubleshooting

### Vector Database Issues

If you see warnings about vector operations, ensure that:
- The pgvector extension is properly installed
- You're using a compatible PostgreSQL version (12+)
- Your database user has permission to create extensions

### API Usage

The system uses OpenAI's API for both embeddings and the agent system. Ensure your API key has access to:
- Text Embedding models
- GPT models used by the Agents SDK

### Memory Usage

For large contracts, you may need to adjust the `max_file_size_mb` parameter in the PDFProcessor class.

## License

[MIT License](LICENSE) 