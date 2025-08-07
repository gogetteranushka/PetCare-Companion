import os
import logging
import streamlit as st
from typing import Tuple
# load_dotenv is only needed for local development
from dotenv import load_dotenv

# When running on your computer, this will load variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Universal API Key Loading ---
TOGETHER_API_KEY = None
TAVILY_API_KEY = None

try:
    # This part is for Streamlit Cloud. It tries to read from the online secrets manager.
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    logger.info("API keys loaded from Streamlit secrets.")
except:
    # This part runs if the app is NOT on Streamlit Cloud (i.e., on your local machine).
    # It falls back to reading from your .env file.
    logger.info("Streamlit secrets not found. Loading API keys from .env file.")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# --- The rest of your configuration is unchanged ---
if not TOGETHER_API_KEY:
    logger.warning("TOGETHER_API_KEY not found. Please set it in Streamlit Cloud secrets or your .env file.")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found. Please set it in Streamlit Cloud secrets or your .env file.")

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
    """Validate Together API key"""
    if not TOGETHER_API_KEY:
        return False, "Together API key not found. Please set it in your .env file or Streamlit Cloud secrets."
    return True, "API key present"