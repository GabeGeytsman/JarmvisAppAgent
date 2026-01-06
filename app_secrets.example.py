# =============================================================================
# SECRETS TEMPLATE - Copy this to secrets.py and fill in your values
# =============================================================================
# Run: cp secrets.example.py secrets.py
# Then edit secrets.py with your actual API keys

# -----------------------------------------------------------------------------
# LLM API (DeepSeek - cheaper alternative to OpenAI)
# Get your key at: https://platform.deepseek.com/
# -----------------------------------------------------------------------------
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "sk-your-deepseek-key-here"

# -----------------------------------------------------------------------------
# Pinecone Vector Database
# Get your free API key at: https://www.pinecone.io/
# -----------------------------------------------------------------------------
PINECONE_API_KEY = "pcsk_your-pinecone-key-here"

# -----------------------------------------------------------------------------
# Neo4j Graph Database
# Default credentials for local Neo4j installation
# Change the password after first login at http://localhost:7474
# -----------------------------------------------------------------------------
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"

# -----------------------------------------------------------------------------
# LangSmith (Optional - for debugging/monitoring)
# Get your key at: https://smith.langchain.com/
# -----------------------------------------------------------------------------
LANGCHAIN_API_KEY = ""  # Leave empty if not using LangSmith
