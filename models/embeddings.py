from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np
from config.config import EMBEDDING_MODEL_NAME

class EmbeddingModel:
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