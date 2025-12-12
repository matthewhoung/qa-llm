"""
Embedder module - converts text chunks into vector embeddings.
Uses sentence-transformers which runs locally (no API needed).
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class Embedder:
    """Handles text embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name: HuggingFace model name. Default is lightweight and fast.
                       Other options: "all-mpnet-base-v2" (better quality, slower)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query: The search query
            
        Returns:
            numpy array of shape (1, embedding_dimension)
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding
