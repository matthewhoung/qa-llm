"""
VectorStore module - handles vector storage and similarity search using FAISS.
FAISS runs entirely locally, no API needed.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional


class VectorStore:
    """Handles vector storage and retrieval using FAISS."""
    
    def __init__(self, dimension: int):
        """
        Initialize the vector store.
        
        Args:
            dimension: The dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (euclidean)
        self.texts: List[str] = []  # Store original texts for retrieval
        self.metadata: List[dict] = []  # Store metadata (e.g., source file)
    
    def add_texts(
        self, 
        texts: List[str], 
        embeddings: np.ndarray,
        metadata: Optional[List[dict]] = None
    ) -> None:
        """
        Add texts and their embeddings to the store.
        
        Args:
            texts: List of text chunks
            embeddings: Corresponding embeddings as numpy array
            metadata: Optional list of metadata dicts for each text
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for the k most similar texts to the query.
        
        Args:
            query_embedding: The query vector
            k: Number of results to return
            
        Returns:
            List of tuples (text, distance, metadata)
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure k doesn't exceed total vectors
        k = min(k, self.index.ntotal)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k
        )
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                results.append((
                    self.texts[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results
    
    def clear(self) -> None:
        """Clear all stored vectors and texts."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
    
    @property
    def count(self) -> int:
        """Return the number of vectors stored."""
        return self.index.ntotal
