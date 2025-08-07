from sentence_transformers import SentenceTransformer
import torch
from typing import Union
import numpy as np
from config.config import EMBEDDING_MODEL_NAME
from langchain.embeddings.base import Embeddings
from typing import List

class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {e}")
            
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for the given text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {e}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """For LangChain compatibility: embeds a list of documents."""
        embeddings = self.get_embeddings(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """For LangChain compatibility: embeds a single query string."""
        embedding = self.get_embeddings(text)[0]
        return embedding.tolist()