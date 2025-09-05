"""
Indexing module for AutoPatternChecker.
Handles FAISS index creation and management for efficient similarity search.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import faiss
import pickle
from pathlib import Path
import json
from .utils import safe_json_serialize

logger = logging.getLogger(__name__)

class FAISSIndexer:
    """Handles FAISS index creation and management for similarity search."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_type = config.get('index_type', 'flat')  # 'flat', 'ivf', 'ivfpq'
        self.metric = config.get('metric', 'cosine')  # 'cosine', 'l2'
        self.nlist = config.get('nlist', 100)  # For IVF indices
        self.m = config.get('m', 8)  # For PQ
        self.nbits = config.get('nbits', 8)  # For PQ
        self.indices = {}  # Store indices per key
        self.mappings = {}  # Store value mappings per key
        self.metadata = {}  # Store metadata per key
        
    def create_index(self, embeddings: np.ndarray, dimension: int) -> faiss.Index:
        """
        Create a FAISS index based on configuration.
        
        Args:
            embeddings: Embeddings array
            dimension: Embedding dimension
        
        Returns:
            FAISS index
        """
        if self.index_type == 'flat':
            if self.metric == 'cosine':
                # For cosine similarity, we need to normalize embeddings
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            else:
                index = faiss.IndexFlatL2(dimension)
        
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
        
        elif self.index_type == 'ivfpq':
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, self.m, self.nbits)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def build_index_for_key(self, composite_key: str, embeddings: np.ndarray, 
                           values: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build FAISS index for a specific composite key.
        
        Args:
            composite_key: The composite key
            embeddings: Embeddings array
            values: List of corresponding values
            metadata: Optional metadata for values
        
        Returns:
            Dictionary with index information
        """
        if len(embeddings) == 0 or len(values) == 0:
            return {
                'composite_key': composite_key,
                'index_type': self.index_type,
                'dimension': 0,
                'count': 0,
                'status': 'empty'
            }
        
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            normalized_embeddings = embeddings.astype(np.float32)
        
        # Create index
        index = self.create_index(normalized_embeddings, dimension)
        
        # Train index if needed (for IVF indices)
        if hasattr(index, 'is_trained') and not index.is_trained:
            if len(normalized_embeddings) < self.nlist:
                logger.warning(f"Not enough data to train IVF index for {composite_key}, using flat index")
                index = faiss.IndexFlatIP(dimension) if self.metric == 'cosine' else faiss.IndexFlatL2(dimension)
            else:
                index.train(normalized_embeddings)
        
        # Add embeddings to index
        index.add(normalized_embeddings.astype(np.float32))
        
        # Store index and mappings
        self.indices[composite_key] = index
        self.mappings[composite_key] = {
            'values': values,
            'indices': list(range(len(values)))
        }
        
        if metadata:
            self.metadata[composite_key] = metadata
        else:
            self.metadata[composite_key] = {}
        
        logger.info(f"Built {self.index_type} index for {composite_key} with {len(values)} vectors")
        
        return {
            'composite_key': composite_key,
            'index_type': self.index_type,
            'dimension': dimension,
            'count': len(values),
            'status': 'success'
        }
    
    def search_similar(self, composite_key: str, query_embedding: np.ndarray, 
                      top_k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        
        Args:
            composite_key: The composite key to search in
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Optional similarity threshold
        
        Returns:
            List of similar results with metadata
        """
        if composite_key not in self.indices:
            logger.warning(f"No index found for key: {composite_key}")
            return []
        
        index = self.indices[composite_key]
        mapping = self.mappings[composite_key]
        
        # Normalize query embedding for cosine similarity
        if self.metric == 'cosine':
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = index.search(query_embedding, min(top_k, len(mapping['values'])))
        
        # Format results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            # Convert similarity based on metric
            if self.metric == 'cosine':
                sim_score = float(similarity)
            else:  # L2 distance
                sim_score = 1.0 / (1.0 + float(similarity))  # Convert distance to similarity
            
            # Apply threshold if specified
            if threshold is not None and sim_score < threshold:
                continue
            
            result = {
                'value': mapping['values'][idx],
                'similarity': sim_score,
                'index': int(idx),
                'rank': i + 1
            }
            
            # Add metadata if available
            if composite_key in self.metadata and idx < len(self.metadata[composite_key]):
                result['metadata'] = self.metadata[composite_key].get(str(idx), {})
            
            results.append(result)
        
        return results
    
    def build_all_indices(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build FAISS indices for all keys.
        
        Args:
            embeddings_data: Dictionary with embeddings data for all keys
        
        Returns:
            Dictionary with index building results
        """
        results = {}
        
        for composite_key, data in embeddings_data.items():
            embeddings = data.get('embeddings', np.array([]))
            values = data.get('values', [])
            metadata = data.get('metadata', {})
            
            result = self.build_index_for_key(composite_key, embeddings, values, metadata)
            results[composite_key] = result
        
        logger.info(f"Built indices for {len(results)} keys")
        return results
    
    def save_indices(self, base_path: str) -> None:
        """
        Save all indices and mappings to disk.
        
        Args:
            base_path: Base path for saving indices
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save each index
        for composite_key, index in self.indices.items():
            # Create safe filename
            safe_key = composite_key.replace('/', '_').replace('\\', '_').replace(':', '_')
            index_path = base_path / f"{safe_key}.index"
            
            # Save FAISS index
            faiss.write_index(index, str(index_path))
            
            # Save mappings
            mapping_path = base_path / f"{safe_key}_mapping.pkl"
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.mappings[composite_key], f)
            
            # Save metadata
            if composite_key in self.metadata:
                metadata_path = base_path / f"{safe_key}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata[composite_key], f, indent=2, default=safe_json_serialize)
        
        # Save index configuration
        config_path = base_path / "index_config.json"
        config = {
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'm': self.m,
            'nbits': self.nbits,
            'keys': list(self.indices.keys())
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {len(self.indices)} indices to {base_path}")
    
    def load_indices(self, base_path: str) -> None:
        """
        Load all indices and mappings from disk.
        
        Args:
            base_path: Base path for loading indices
        """
        base_path = Path(base_path)
        
        # Load index configuration
        config_path = base_path / "index_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.index_type = config.get('index_type', self.index_type)
            self.metric = config.get('metric', self.metric)
            self.nlist = config.get('nlist', self.nlist)
            self.m = config.get('m', self.m)
            self.nbits = config.get('nbits', self.nbits)
        
        # Load each index
        for key in config.get('keys', []):
            safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
            index_path = base_path / f"{safe_key}.index"
            mapping_path = base_path / f"{safe_key}_mapping.pkl"
            metadata_path = base_path / f"{safe_key}_metadata.json"
            
            if index_path.exists() and mapping_path.exists():
                # Load FAISS index
                index = faiss.read_index(str(index_path))
                self.indices[key] = index
                
                # Load mappings
                with open(mapping_path, 'rb') as f:
                    self.mappings[key] = pickle.load(f)
                
                # Load metadata if exists
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata[key] = json.load(f)
                else:
                    self.metadata[key] = {}
        
        logger.info(f"Loaded {len(self.indices)} indices from {base_path}")
    
    def get_index_info(self, composite_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about indices.
        
        Args:
            composite_key: Optional specific key to get info for
        
        Returns:
            Dictionary with index information
        """
        if composite_key:
            if composite_key not in self.indices:
                return {'error': f'No index found for key: {composite_key}'}
            
            index = self.indices[composite_key]
            mapping = self.mappings[composite_key]
            
            return {
                'composite_key': composite_key,
                'index_type': self.index_type,
                'metric': self.metric,
                'count': len(mapping['values']),
                'dimension': index.d,
                'is_trained': getattr(index, 'is_trained', True)
            }
        else:
            return {
                'total_indices': len(self.indices),
                'index_type': self.index_type,
                'metric': self.metric,
                'keys': list(self.indices.keys())
            }
    
    def update_index(self, composite_key: str, new_embeddings: np.ndarray, 
                    new_values: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update an existing index with new data.
        
        Args:
            composite_key: The composite key
            new_embeddings: New embeddings to add
            new_values: New values to add
            metadata: Optional metadata for new values
        
        Returns:
            Update result
        """
        if composite_key not in self.indices:
            return {'error': f'No index found for key: {composite_key}'}
        
        # This is a simplified update - in practice, you might want to rebuild the index
        # for better performance with large updates
        logger.warning("Index update not fully implemented - consider rebuilding for large updates")
        
        return {
            'composite_key': composite_key,
            'status': 'update_not_implemented',
            'message': 'Consider rebuilding index for updates'
        }
    
    def remove_index(self, composite_key: str) -> bool:
        """
        Remove an index for a specific key.
        
        Args:
            composite_key: The composite key to remove
        
        Returns:
            True if removed successfully
        """
        if composite_key in self.indices:
            del self.indices[composite_key]
            del self.mappings[composite_key]
            if composite_key in self.metadata:
                del self.metadata[composite_key]
            logger.info(f"Removed index for key: {composite_key}")
            return True
        else:
            logger.warning(f"No index found for key: {composite_key}")
            return False