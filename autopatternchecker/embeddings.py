"""
Embeddings module for AutoPatternChecker.
Handles embedding generation and batch processing for semantic similarity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles embedding generation for semantic similarity analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.batch_size = config.get('embedding_batch_size', 64)
        self.device = self._get_device()
        self.model = None
        self.embedding_cache = {}
        
    def _get_device(self) -> str:
        """Determine the best device for embedding generation."""
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Initialized embedding model: {self.model_name} on {self.device}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise
    
    def generate_embeddings(self, texts: List[str], 
                           cache_key: Optional[str] = None,
                           show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            cache_key: Optional cache key for storing/retrieving embeddings
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Check cache first
        if cache_key and cache_key in self.embedding_cache:
            logger.info(f"Using cached embeddings for {cache_key}")
            return self.embedding_cache[cache_key]
        
        # Initialize model if needed
        self._initialize_model()
        
        # Clean and prepare texts
        cleaned_texts = [str(text).strip() if pd.notna(text) else "" for text in texts]
        
        # Generate embeddings in batches
        embeddings = []
        
        if show_progress:
            iterator = tqdm(range(0, len(cleaned_texts), self.batch_size), 
                          desc="Generating embeddings")
        else:
            iterator = range(0, len(cleaned_texts), self.batch_size)
        
        for i in iterator:
            batch_texts = cleaned_texts[i:i + self.batch_size]
            if batch_texts:
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i}: {e}")
                    # Create zero embeddings for failed batch
                    zero_embeddings = np.zeros((len(batch_texts), 384))  # Default dimension
                    embeddings.append(zero_embeddings)
        
        # Combine all embeddings
        if embeddings:
            all_embeddings = np.vstack(embeddings)
        else:
            all_embeddings = np.array([])
        
        # Cache if requested
        if cache_key:
            self.embedding_cache[cache_key] = all_embeddings
        
        logger.info(f"Generated {len(all_embeddings)} embeddings with dimension {all_embeddings.shape[1] if len(all_embeddings) > 0 else 0}")
        return all_embeddings
    
    def generate_embeddings_for_key(self, composite_key: str, values: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for values of a specific key.
        
        Args:
            composite_key: The composite key
            values: List of values to embed
        
        Returns:
            Dictionary with embeddings and metadata
        """
        if not values:
            return {
                'composite_key': composite_key,
                'embeddings': np.array([]),
                'values': [],
                'model_name': self.model_name,
                'dimension': 0
            }
        
        # Generate embeddings
        embeddings = self.generate_embeddings(values, cache_key=f"{composite_key}_embeddings")
        
        return {
            'composite_key': composite_key,
            'embeddings': embeddings,
            'values': values,
            'model_name': self.model_name,
            'dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'count': len(values)
        }
    
    def generate_embeddings_for_all_keys(self, key_values: Dict[str, List[str]], 
                                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Generate embeddings for all keys.
        
        Args:
            key_values: Dictionary mapping composite keys to lists of values
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary with embeddings for all keys
        """
        all_embeddings = {}
        
        if show_progress:
            iterator = tqdm(key_values.items(), desc="Processing keys for embeddings")
        else:
            iterator = key_values.items()
        
        for composite_key, values in iterator:
            if values:
                result = self.generate_embeddings_for_key(composite_key, values)
                all_embeddings[composite_key] = result
            else:
                logger.warning(f"No values found for key: {composite_key}")
        
        logger.info(f"Generated embeddings for {len(all_embeddings)} keys")
        return all_embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
        
        Returns:
            Similarity matrix
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Normalize embeddings
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def find_similar_pairs(self, embeddings: np.ndarray, values: List[str], 
                          threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find pairs of similar values based on embedding similarity.
        
        Args:
            embeddings: Numpy array of embeddings
            values: List of corresponding values
            threshold: Similarity threshold
        
        Returns:
            List of similar pairs with metadata
        """
        if len(embeddings) == 0 or len(values) == 0:
            return []
        
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        similar_pairs = []
        
        # Find pairs above threshold (excluding diagonal)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    similar_pairs.append({
                        'value1': values[i],
                        'value2': values[j],
                        'similarity': float(similarity),
                        'index1': i,
                        'index2': j
                    })
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_pairs
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray,
                         candidate_values: List[str],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar values to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embeddings matrix
            candidate_values: List of candidate values
            top_k: Number of top results to return
        
        Returns:
            List of most similar values with metadata
        """
        if len(candidate_embeddings) == 0 or len(candidate_values) == 0:
            return []
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(candidate_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'value': candidate_values[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results
    
    def compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics for a set of embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
        
        Returns:
            Dictionary with embedding statistics
        """
        if len(embeddings) == 0:
            return {
                'count': 0,
                'dimension': 0,
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'min_norm': 0.0,
                'max_norm': 0.0
            }
        
        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            'count': len(embeddings),
            'dimension': embeddings.shape[1],
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms))
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the embedding cache."""
        return {
            'cache_size': len(self.embedding_cache),
            'cached_keys': list(self.embedding_cache.keys()),
            'model_name': self.model_name,
            'device': self.device
        }
    
    def save_embeddings(self, embeddings_data: Dict[str, Any], filepath: str) -> None:
        """
        Save embeddings data to file.
        
        Args:
            embeddings_data: Dictionary with embeddings data
            filepath: Path to save the data
        """
        import pickle
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, Any]:
        """
        Load embeddings data from file.
        
        Args:
            filepath: Path to load the data from
        
        Returns:
            Dictionary with embeddings data
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings_data