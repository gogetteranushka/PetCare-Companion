import os
import logging
from typing import Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.warning("TOGETHER_API_KEY not found in environment variables")

# Tavily API Key for web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found in environment variables")

# Embedding Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# App Settings
APP_TITLE = os.getenv("APP_TITLE", "PetCare Companion")

# LLM Settings
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Vector DB Settings
VECTOR_DIMENSION = 1024  # BGE-large dimension
COLLECTION_NAME = "documents"

# Response settings
RESPONSE_MODES = {
    "concise": "Provide a short, summarized answer",
    "detailed": "Provide a detailed, comprehensive explanation"
}

# Pet species options
PET_SPECIES = [
    "All species",
    "Dogs",
    "Cats", 
    "Small mammals",
    "Birds",
    "Reptiles",
    "Fish"
]

# Function to validate API keys
def validate_together_api_key() -> Tuple[bool, str]:
    """Validate Together API key
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not TOGETHER_API_KEY:
        return False, "Together API key not found in .env file"
    return True, "API key present"