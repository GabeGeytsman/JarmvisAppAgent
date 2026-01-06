# =============================================================================
# AppAgentX Configuration
# =============================================================================
# Sensitive values (API keys, passwords) are loaded from secrets.py
# Copy secrets.example.py to secrets.py and fill in your values

try:
    from app_secrets import (
        LLM_BASE_URL,
        LLM_API_KEY,
        PINECONE_API_KEY,
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        LANGCHAIN_API_KEY,
    )
except ImportError:
    raise ImportError(
        "app_secrets.py not found! Copy app_secrets.example.py to app_secrets.py and fill in your API keys."
    )

# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
LLM_MODEL = "gpt-4o"  # OpenAI GPT-4o (vision-capable)
LLM_MAX_TOKEN = 1500
LLM_REQUEST_TIMEOUT = 500
LLM_MAX_RETRIES = 3

# -----------------------------------------------------------------------------
# LangChain/LangSmith Configuration
# -----------------------------------------------------------------------------
LANGCHAIN_TRACING_V2 = "false"  # Set to "true" to enable LangSmith tracing
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "AppAgentX"

# -----------------------------------------------------------------------------
# Neo4j Configuration (credentials from secrets.py)
# -----------------------------------------------------------------------------
Neo4j_URI = NEO4J_URI
Neo4j_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# -----------------------------------------------------------------------------
# Backend Services Configuration
# These run locally - no secrets needed
# -----------------------------------------------------------------------------
Feature_URI = "http://127.0.0.1:8001"  # Image feature extraction service
Omni_URI = "http://127.0.0.1:8000"     # OmniParser screen parsing service
