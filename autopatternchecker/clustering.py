"""
Clustering module for AutoPatternChecker.
Handles sub-format detection within composite keys using HDBSCAN and other clustering methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import hdbscan
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """Handles clustering analysis for detecting sub-formats within composite keys."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tfidf_params = {
            'ngram_range': tuple(config.get('tfidf_ngram_range', [2, 4])),
            'max_features': config.get('tfidf_max_features', 2000),
            'min_df': config.get('tfidf_min_df', 1),
            'max_df': config.get('tfidf_max_df', 0.95)
        }
        self.hdbscan_params = {
            'min_cluster_size': config.get('hdbscan_min_cluster_size', 5),
            'min_samples': config.get('hdbscan_min_samples', 2),
            'metric': 'cosine'
        }
        self.kmeans_params = {
            'n_clusters': config.get('kmeans_n_clusters', 8),
            'n_init': config.get('kmeans_n_init', 5),
            'random_state': 42
        }
        self.embedding_model = None
        self.use_embeddings = config.get('use_embeddings', False)
        
    def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer model for embeddings."""
        if self.embedding_model is None:
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Initialized embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.use_embeddings = False
    
    def cluster_key_values(self, composite_key: str, values: List[str], 
                          key_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster values for a specific composite key to detect sub-formats.
        
        Args:
            composite_key: The composite key
            values: List of values to cluster
            key_profile: Profile information for the key
        
        Returns:
            Dictionary with clustering results
        """
        if len(values) < 5:  # Not enough data for clustering
            return {
                'composite_key': composite_key,
                'n_clusters': 0,
                'clusters': [],
                'method': 'insufficient_data',
                'silhouette_score': 0.0
            }
        
        # Determine clustering method based on key characteristics
        if key_profile.get('is_numeric_unit', False):
            return self._cluster_numeric_unit_values(composite_key, values, key_profile)
        elif key_profile.get('is_free_text', False) and self.use_embeddings:
            return self._cluster_with_embeddings(composite_key, values, key_profile)
        else:
            return self._cluster_with_tfidf(composite_key, values, key_profile)
    
    def _cluster_with_tfidf(self, composite_key: str, values: List[str], 
                           key_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster values using TF-IDF vectorization."""
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(**self.tfidf_params)
            tfidf_matrix = vectorizer.fit_transform(values)
            
            # Try HDBSCAN first
            clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
            cluster_labels = clusterer.fit_predict(tfidf_matrix)
            
            # If HDBSCAN doesn't find clusters, try KMeans
            if len(set(cluster_labels)) <= 1:
                logger.info(f"HDBSCAN found no clusters for {composite_key}, trying KMeans")
                kmeans = MiniBatchKMeans(**self.kmeans_params)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                method = 'kmeans'
            else:
                method = 'hdbscan'
            
            # Analyze clusters
            clusters = self._analyze_clusters(values, cluster_labels, vectorizer)
            
            # Calculate silhouette score if possible
            silhouette_score = self._calculate_silhouette_score(tfidf_matrix, cluster_labels)
            
            return {
                'composite_key': composite_key,
                'n_clusters': len(set(cluster_labels)),
                'clusters': clusters,
                'method': method,
                'silhouette_score': silhouette_score,
                'vectorizer_params': self.tfidf_params
            }
            
        except Exception as e:
            logger.error(f"Error clustering with TF-IDF for {composite_key}: {e}")
            return {
                'composite_key': composite_key,
                'n_clusters': 0,
                'clusters': [],
                'method': 'error',
                'error': str(e)
            }
    
    def _cluster_with_embeddings(self, composite_key: str, values: List[str], 
                                key_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster values using sentence embeddings."""
        try:
            # Initialize embedding model if needed
            self._initialize_embedding_model()
            if self.embedding_model is None:
                return self._cluster_with_tfidf(composite_key, values, key_profile)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(values, batch_size=32)
            
            # Cluster using HDBSCAN
            clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # If HDBSCAN doesn't find clusters, try KMeans
            if len(set(cluster_labels)) <= 1:
                logger.info(f"HDBSCAN found no clusters for {composite_key}, trying KMeans")
                kmeans = MiniBatchKMeans(**self.kmeans_params)
                cluster_labels = kmeans.fit_predict(embeddings)
                method = 'kmeans'
            else:
                method = 'hdbscan'
            
            # Analyze clusters
            clusters = self._analyze_clusters(values, cluster_labels)
            
            # Calculate silhouette score
            silhouette_score = self._calculate_silhouette_score(embeddings, cluster_labels)
            
            return {
                'composite_key': composite_key,
                'n_clusters': len(set(cluster_labels)),
                'clusters': clusters,
                'method': method,
                'silhouette_score': silhouette_score,
                'embedding_model': self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            }
            
        except Exception as e:
            logger.error(f"Error clustering with embeddings for {composite_key}: {e}")
            return self._cluster_with_tfidf(composite_key, values, key_profile)
    
    def _cluster_numeric_unit_values(self, composite_key: str, values: List[str], 
                                   key_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized clustering for numeric + unit values."""
        try:
            # Parse numeric and unit components
            parsed_values = []
            for value in values:
                parsed = self._parse_numeric_unit(value)
                parsed_values.append(parsed)
            
            # Extract numeric values for clustering
            numeric_values = [p['numeric'] for p in parsed_values if p['numeric'] is not None]
            
            if len(numeric_values) < 5:
                return {
                    'composite_key': composite_key,
                    'n_clusters': 0,
                    'clusters': [],
                    'method': 'insufficient_numeric_data'
                }
            
            # Cluster based on numeric ranges
            numeric_array = np.array(numeric_values).reshape(-1, 1)
            
            # Use KMeans for numeric clustering
            kmeans = MiniBatchKMeans(n_clusters=min(5, len(numeric_values)//2), random_state=42)
            cluster_labels = kmeans.fit_predict(numeric_array)
            
            # Group by clusters and units
            clusters = self._analyze_numeric_clusters(values, parsed_values, cluster_labels)
            
            return {
                'composite_key': composite_key,
                'n_clusters': len(set(cluster_labels)),
                'clusters': clusters,
                'method': 'numeric_unit',
                'silhouette_score': 0.0  # Not meaningful for this method
            }
            
        except Exception as e:
            logger.error(f"Error clustering numeric values for {composite_key}: {e}")
            return {
                'composite_key': composite_key,
                'n_clusters': 0,
                'clusters': [],
                'method': 'error',
                'error': str(e)
            }
    
    def _parse_numeric_unit(self, value: str) -> Dict[str, Any]:
        """Parse a value into numeric and unit components."""
        import re
        
        # Pattern for numeric + unit
        pattern = r'^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$'
        match = re.match(pattern, str(value).strip())
        
        if match:
            numeric = float(match.group(1))
            unit = match.group(2)
            return {
                'numeric': numeric,
                'unit': unit,
                'original': value
            }
        else:
            return {
                'numeric': None,
                'unit': None,
                'original': value
            }
    
    def _analyze_clusters(self, values: List[str], cluster_labels: np.ndarray, 
                         vectorizer: Optional[TfidfVectorizer] = None) -> List[Dict[str, Any]]:
        """Analyze clustering results and extract cluster information."""
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in HDBSCAN
                continue
                
            cluster_mask = cluster_labels == label
            cluster_values = [values[i] for i in range(len(values)) if cluster_mask[i]]
            
            if len(cluster_values) == 0:
                continue
            
            # Generate signature for this cluster
            from .utils import generate_signature
            signatures = [generate_signature(val) for val in cluster_values]
            most_common_sig = max(set(signatures), key=signatures.count)
            
            # Generate cluster regex
            from .utils import signature_to_regex
            cluster_regex = signature_to_regex(most_common_sig)
            
            # Get top terms if using TF-IDF
            top_terms = []
            if vectorizer is not None:
                try:
                    cluster_indices = [i for i, mask in enumerate(cluster_mask) if mask]
                    if cluster_indices:
                        cluster_tfidf = vectorizer.transform(cluster_values)
                        feature_names = vectorizer.get_feature_names_out()
                        mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
                        top_indices = np.argsort(mean_scores)[-5:][::-1]
                        top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
                except Exception as e:
                    logger.warning(f"Error extracting top terms: {e}")
            
            clusters.append({
                'cluster_id': int(label),
                'size': len(cluster_values),
                'pattern_signature': most_common_sig,
                'cluster_regex': cluster_regex,
                'example_values': cluster_values[:3],
                'top_terms': top_terms,
                'coverage_pct': round((len(cluster_values) / len(values)) * 100, 2)
            })
        
        return clusters
    
    def _analyze_numeric_clusters(self, values: List[str], parsed_values: List[Dict[str, Any]], 
                                 cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze clusters for numeric + unit values."""
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_values = [values[i] for i in range(len(values)) if cluster_mask[i]]
            cluster_parsed = [parsed_values[i] for i in range(len(parsed_values)) if cluster_mask[i]]
            
            if len(cluster_values) == 0:
                continue
            
            # Analyze numeric range
            numeric_values = [p['numeric'] for p in cluster_parsed if p['numeric'] is not None]
            units = [p['unit'] for p in cluster_parsed if p['unit'] is not None]
            
            numeric_stats = {}
            if numeric_values:
                numeric_stats = {
                    'min': float(min(numeric_values)),
                    'max': float(max(numeric_values)),
                    'mean': float(np.mean(numeric_values)),
                    'std': float(np.std(numeric_values))
                }
            
            # Most common unit
            most_common_unit = max(set(units), key=units.count) if units else None
            
            clusters.append({
                'cluster_id': int(label),
                'size': len(cluster_values),
                'numeric_stats': numeric_stats,
                'most_common_unit': most_common_unit,
                'example_values': cluster_values[:3],
                'coverage_pct': round((len(cluster_values) / len(values)) * 100, 2)
            })
        
        return clusters
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering evaluation."""
        try:
            from sklearn.metrics import silhouette_score
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Remove noise points (-1) for silhouette calculation
            valid_mask = labels != -1
            if valid_mask.sum() < 2:
                return 0.0
            
            valid_features = features[valid_mask] if hasattr(features, '__getitem__') else features[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return 0.0
            
            return float(silhouette_score(valid_features, valid_labels))
        except Exception as e:
            logger.warning(f"Error calculating silhouette score: {e}")
            return 0.0
    
    def cluster_all_keys(self, key_profiles: Dict[str, Any], 
                        values_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Cluster values for all keys.
        
        Args:
            key_profiles: Dictionary of key profiles
            values_data: Dictionary mapping composite keys to lists of values
        
        Returns:
            Dictionary with clustering results for all keys
        """
        clustering_results = {}
        
        for composite_key, profile in key_profiles.items():
            if composite_key in values_data:
                values = values_data[composite_key]
                result = self.cluster_key_values(composite_key, values, profile)
                clustering_results[composite_key] = result
            else:
                logger.warning(f"No values found for key: {composite_key}")
        
        logger.info(f"Completed clustering for {len(clustering_results)} keys")
        return clustering_results
    
    def suggest_clustering_improvements(self, clustering_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Suggest improvements for clustering results."""
        suggestions = {}
        
        for composite_key, result in clustering_results.items():
            key_suggestions = []
            
            if result.get('n_clusters', 0) == 0:
                key_suggestions.append("No clusters found - consider different parameters or methods")
            
            if result.get('silhouette_score', 0) < 0.3:
                key_suggestions.append("Low silhouette score - clusters may not be well-separated")
            
            if result.get('method') == 'kmeans' and result.get('n_clusters', 0) > 0:
                key_suggestions.append("Using KMeans fallback - consider tuning HDBSCAN parameters")
            
            # Check for clusters with very few samples
            clusters = result.get('clusters', [])
            small_clusters = [c for c in clusters if c.get('size', 0) < 3]
            if small_clusters:
                key_suggestions.append(f"Found {len(small_clusters)} clusters with <3 samples")
            
            if key_suggestions:
                suggestions[composite_key] = key_suggestions
        
        return suggestions