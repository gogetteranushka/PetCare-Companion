# models/embeddings.py

from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np
from langchain.embeddings.base import Embeddings
# --- THIS IS THE LINE YOU NEED TO ADD ---
from config.config import EMBEDDING_MODEL_NAME

class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {e}")
            
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """The core embedding generation logic."""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {e}")

    # --- Methods for LangChain Compatibility ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """For LangChain: embeds a list of documents."""
        embeddings = self.get_embeddings(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """For LangChain: embeds a single query string."""
        embedding = self.get_embeddings(text)[0]
        return embedding.tolist()