"""
Model training module for AutoPatternChecker.
Handles fine-tuning of embedding models and optimization of clustering parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Embedding libraries
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and fine-tuning for AutoPatternChecker."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_data = {}
        self.trained_models = {}
        self.training_history = {}
        
    def prepare_training_data(self, key_profiles: Dict[str, Any], 
                            processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare training data for different model types.
        
        Args:
            key_profiles: Dictionary of key profiles
            processed_df: Processed dataframe with values
        
        Returns:
            Dictionary with prepared training data
        """
        logger.info("Preparing training data...")
        
        training_data = {
            'clustering_data': {},
            'embedding_data': {},
            'normalization_data': {},
            'pattern_data': {}
        }
        
        # Prepare clustering training data
        for composite_key, profile in key_profiles.items():
            if profile.get('count', 0) < 10:  # Skip keys with too few samples
                continue
                
            # Get values for this key
            key_values = processed_df[processed_df['__composite_key__'] == composite_key]['value'].dropna().tolist()
            
            if len(key_values) < 10:
                continue
            
            # Prepare data for different training tasks
            training_data['clustering_data'][composite_key] = {
                'values': key_values,
                'features': self._extract_clustering_features(key_values),
                'labels': self._generate_clustering_labels(key_values, profile)
            }
            
            training_data['embedding_data'][composite_key] = {
                'values': key_values,
                'pairs': self._generate_embedding_pairs(key_values),
                'triplets': self._generate_embedding_triplets(key_values)
            }
            
            training_data['normalization_data'][composite_key] = {
                'original_values': key_values,
                'normalized_values': self._generate_normalized_examples(key_values),
                'rules': profile.get('normalization_rules', [])
            }
            
            training_data['pattern_data'][composite_key] = {
                'values': key_values,
                'signatures': [self._generate_signature(val) for val in key_values],
                'patterns': profile.get('top_signatures', [])
            }
        
        self.training_data = training_data
        logger.info(f"Prepared training data for {len(training_data['clustering_data'])} keys")
        return training_data
    
    def _extract_clustering_features(self, values: List[str]) -> np.ndarray:
        """Extract features for clustering."""
        # Use TF-IDF features
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=1000,
            min_df=1,
            max_df=0.95
        )
        
        features = vectorizer.fit_transform(values).toarray()
        return features
    
    def _generate_clustering_labels(self, values: List[str], profile: Dict[str, Any]) -> np.ndarray:
        """Generate clustering labels based on signatures."""
        signatures = [self._generate_signature(val) for val in values]
        unique_signatures = list(set(signatures))
        
        # Create labels based on signature similarity
        labels = np.zeros(len(values))
        for i, sig in enumerate(signatures):
            labels[i] = unique_signatures.index(sig)
        
        return labels
    
    def _generate_embedding_pairs(self, values: List[str]) -> List[Tuple[str, str, float]]:
        """Generate positive and negative pairs for embedding training."""
        pairs = []
        
        # Generate positive pairs (similar values)
        for i, val1 in enumerate(values):
            for j, val2 in enumerate(values[i+1:], i+1):
                similarity = self._calculate_text_similarity(val1, val2)
                if similarity > 0.8:  # High similarity
                    pairs.append((val1, val2, 1.0))
                elif similarity < 0.3:  # Low similarity
                    pairs.append((val1, val2, 0.0))
        
        return pairs
    
    def _generate_embedding_triplets(self, values: List[str]) -> List[Tuple[str, str, str]]:
        """Generate triplets for contrastive learning."""
        triplets = []
        
        # Group values by signature
        signature_groups = {}
        for val in values:
            sig = self._generate_signature(val)
            if sig not in signature_groups:
                signature_groups[sig] = []
            signature_groups[sig].append(val)
        
        # Generate triplets: anchor, positive, negative
        for sig, group_values in signature_groups.items():
            if len(group_values) < 2:
                continue
                
            # Find values from different signature groups
            other_groups = [vals for s, vals in signature_groups.items() if s != sig]
            if not other_groups:
                continue
            
            for anchor in group_values:
                for positive in group_values:
                    if anchor != positive:
                        # Select negative from different group
                        negative_group = other_groups[0]
                        negative = negative_group[0]
                        triplets.append((anchor, positive, negative))
        
        return triplets
    
    def _generate_normalized_examples(self, values: List[str]) -> List[Tuple[str, str]]:
        """Generate normalized examples for training."""
        examples = []
        
        for val in values:
            # Apply basic normalization
            normalized = val.strip().lower()
            if normalized != val:
                examples.append((val, normalized))
        
        return examples
    
    def _generate_signature(self, value: str) -> str:
        """Generate signature for a value."""
        if not value:
            return ""
        
        signature = []
        current_char = None
        count = 0
        
        for char in str(value):
            if char.isdigit():
                char_type = 'D'
            elif char.isalpha():
                char_type = 'L'
            elif char.isspace():
                char_type = '_'
            else:
                char_type = 'O'
            
            if char_type == current_char:
                count += 1
            else:
                if current_char is not None:
                    if count > 1:
                        signature.append(f"{current_char}{count}")
                    else:
                        signature.append(current_char)
                current_char = char_type
                count = 1
        
        if current_char is not None:
            if count > 1:
                signature.append(f"{current_char}{count}")
            else:
                signature.append(current_char)
        
        return "".join(signature)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple metrics."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def train_clustering_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train and optimize clustering models for each key.
        
        Args:
            training_data: Prepared training data
        
        Returns:
            Dictionary with trained clustering models
        """
        logger.info("Training clustering models...")
        
        trained_models = {}
        
        for composite_key, data in training_data['clustering_data'].items():
            logger.info(f"Training clustering for key: {composite_key}")
            
            features = data['features']
            labels = data['labels']
            
            if len(features) < 10:
                logger.warning(f"Not enough data for clustering: {composite_key}")
                continue
            
            # Train HDBSCAN with parameter optimization
            hdbscan_model = self._optimize_hdbscan(features, labels)
            
            # Train KMeans with optimal cluster number
            kmeans_model = self._optimize_kmeans(features, labels)
            
            # Select best model
            best_model = self._select_best_clustering_model(
                features, labels, hdbscan_model, kmeans_model
            )
            
            trained_models[composite_key] = {
                'model': best_model,
                'hdbscan': hdbscan_model,
                'kmeans': kmeans_model,
                'features': features,
                'labels': labels,
                'n_clusters': len(set(labels)) if hasattr(labels, '__len__') else 0
            }
        
        self.trained_models['clustering'] = trained_models
        logger.info(f"Trained clustering models for {len(trained_models)} keys")
        return trained_models
    
    def _optimize_hdbscan(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters."""
        param_grid = {
            'min_cluster_size': [3, 5, 10, 15],
            'min_samples': [1, 2, 3, 5],
            'metric': ['euclidean', 'cosine', 'manhattan']
        }
        
        best_score = -1
        best_params = None
        best_model = None
        
        for min_cluster_size in param_grid['min_cluster_size']:
            for min_samples in param_grid['min_samples']:
                for metric in param_grid['metric']:
                    try:
                        model = HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric=metric
                        )
                        
                        cluster_labels = model.fit_predict(features)
                        
                        if len(set(cluster_labels)) > 1:
                            score = silhouette_score(features, cluster_labels)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'metric': metric
                                }
                                best_model = model
                    except Exception as e:
                        logger.warning(f"HDBSCAN optimization error: {e}")
                        continue
        
        return {
            'model': best_model,
            'params': best_params,
            'score': best_score
        }
    
    def _optimize_kmeans(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Optimize KMeans parameters."""
        n_clusters_range = range(2, min(20, len(features) // 2))
        
        best_score = -1
        best_params = None
        best_model = None
        
        for n_clusters in n_clusters_range:
            try:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = model.fit_predict(features)
                
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(features, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_clusters': n_clusters}
                        best_model = model
            except Exception as e:
                logger.warning(f"KMeans optimization error: {e}")
                continue
        
        return {
            'model': best_model,
            'params': best_params,
            'score': best_score
        }
    
    def _select_best_clustering_model(self, features: np.ndarray, labels: np.ndarray,
                                    hdbscan_result: Dict[str, Any], 
                                    kmeans_result: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best clustering model based on performance."""
        hdbscan_score = hdbscan_result.get('score', -1)
        kmeans_score = kmeans_result.get('score', -1)
        
        if hdbscan_score > kmeans_score:
            return {
                'type': 'hdbscan',
                'model': hdbscan_result['model'],
                'params': hdbscan_result['params'],
                'score': hdbscan_score
            }
        else:
            return {
                'type': 'kmeans',
                'model': kmeans_result['model'],
                'params': kmeans_result['params'],
                'score': kmeans_score
            }
    
    def train_embedding_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fine-tune embedding models for better semantic similarity.
        
        Args:
            training_data: Prepared training data
        
        Returns:
            Dictionary with trained embedding models
        """
        logger.info("Training embedding models...")
        
        # Prepare training examples
        all_pairs = []
        all_triplets = []
        
        for composite_key, data in training_data['embedding_data'].items():
            all_pairs.extend(data['pairs'])
            all_triplets.extend(data['triplets'])
        
        if not all_pairs and not all_triplets:
            logger.warning("No training data available for embedding training")
            return {}
        
        # Load base model
        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        model = SentenceTransformer(model_name)
        
        # Prepare training examples
        train_examples = []
        
        # Add pairs
        for anchor, positive, score in all_pairs[:1000]:  # Limit for memory
            train_examples.append(InputExample(texts=[anchor, positive], label=float(score)))
        
        # Add triplets
        for anchor, positive, negative in all_triplets[:1000]:  # Limit for memory
            train_examples.append(InputExample(texts=[anchor, positive, negative]))
        
        if not train_examples:
            logger.warning("No valid training examples for embedding training")
            return {}
        
        # Split data
        train_examples, val_examples = train_test_split(
            train_examples, test_size=0.2, random_state=42
        )
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss function
        if all_triplets:
            train_loss = losses.TripletLoss(model)
        else:
            train_loss = losses.CosineSimilarityLoss(model)
        
        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            output_path=f'/tmp/finetuned_embedding_model',
            save_best_model=True
        )
        
        # Load the best model
        trained_model = SentenceTransformer(f'/tmp/finetuned_embedding_model')
        
        self.trained_models['embedding'] = {
            'model': trained_model,
            'base_model': model_name,
            'training_examples': len(train_examples),
            'validation_examples': len(val_examples)
        }
        
        logger.info(f"Trained embedding model with {len(train_examples)} examples")
        return self.trained_models['embedding']
    
    def train_normalization_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train normalization models for better text cleaning.
        
        Args:
            training_data: Prepared training data
        
        Returns:
            Dictionary with trained normalization models
        """
        logger.info("Training normalization models...")
        
        # Prepare training data for normalization
        all_examples = []
        
        for composite_key, data in training_data['normalization_data'].items():
            all_examples.extend(data['normalized_values'])
        
        if not all_examples:
            logger.warning("No normalization training data available")
            return {}
        
        # Train a simple normalization model (this could be enhanced with more sophisticated approaches)
        normalization_rules = self._learn_normalization_rules(all_examples)
        
        self.trained_models['normalization'] = {
            'rules': normalization_rules,
            'training_examples': len(all_examples)
        }
        
        logger.info(f"Trained normalization model with {len(all_examples)} examples")
        return self.trained_models['normalization']
    
    def _learn_normalization_rules(self, examples: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Learn normalization rules from examples."""
        rules = []
        
        # Analyze common transformations
        transformations = {}
        
        for original, normalized in examples:
            if original != normalized:
                # Find the transformation
                if original.lower() == normalized:
                    transformations['to_lowercase'] = transformations.get('to_lowercase', 0) + 1
                elif original.strip() == normalized:
                    transformations['trim_whitespace'] = transformations.get('trim_whitespace', 0) + 1
                elif len(original) > len(normalized):
                    transformations['remove_extra_chars'] = transformations.get('remove_extra_chars', 0) + 1
        
        # Create rules based on frequency
        for transformation, count in transformations.items():
            if count > len(examples) * 0.1:  # If used in more than 10% of cases
                rules.append({
                    'name': transformation,
                    'confidence': count / len(examples),
                    'frequency': count
                })
        
        return rules
    
    def evaluate_models(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trained models on test data.
        
        Args:
            test_data: Test data for evaluation
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating trained models...")
        
        evaluation_results = {
            'clustering': {},
            'embedding': {},
            'normalization': {}
        }
        
        # Evaluate clustering models
        if 'clustering' in self.trained_models:
            for composite_key, model_data in self.trained_models['clustering'].items():
                if composite_key in test_data.get('clustering_data', {}):
                    test_features = test_data['clustering_data'][composite_key]['features']
                    test_labels = test_data['clustering_data'][composite_key]['labels']
                    
                    # Predict clusters
                    predicted_labels = model_data['model']['model'].fit_predict(test_features)
                    
                    # Calculate metrics
                    if len(set(predicted_labels)) > 1:
                        silhouette = silhouette_score(test_features, predicted_labels)
                        ari = adjusted_rand_score(test_labels, predicted_labels)
                        
                        evaluation_results['clustering'][composite_key] = {
                            'silhouette_score': silhouette,
                            'adjusted_rand_index': ari,
                            'n_clusters_predicted': len(set(predicted_labels))
                        }
        
        # Evaluate embedding models
        if 'embedding' in self.trained_models:
            # This would require more sophisticated evaluation
            evaluation_results['embedding'] = {
                'model_type': 'fine_tuned',
                'base_model': self.trained_models['embedding']['base_model']
            }
        
        # Evaluate normalization models
        if 'normalization' in self.trained_models:
            evaluation_results['normalization'] = {
                'rules_learned': len(self.trained_models['normalization']['rules']),
                'training_examples': self.trained_models['normalization']['training_examples']
            }
        
        self.training_history['evaluation'] = evaluation_results
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def save_trained_models(self, output_path: str) -> None:
        """Save trained models to disk."""
        logger.info(f"Saving trained models to {output_path}")
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save clustering models
        if 'clustering' in self.trained_models:
            clustering_path = f'{output_path}/clustering_models.pkl'
            with open(clustering_path, 'wb') as f:
                pickle.dump(self.trained_models['clustering'], f)
            logger.info(f"Saved clustering models to {clustering_path}")
        
        # Save embedding model
        if 'embedding' in self.trained_models:
            embedding_path = f'{output_path}/embedding_model'
            self.trained_models['embedding']['model'].save(embedding_path)
            logger.info(f"Saved embedding model to {embedding_path}")
        
        # Save normalization models
        if 'normalization' in self.trained_models:
            normalization_path = f'{output_path}/normalization_models.json'
            with open(normalization_path, 'w') as f:
                json.dump(self.trained_models['normalization'], f, indent=2)
            logger.info(f"Saved normalization models to {normalization_path}")
        
        # Save training history
        history_path = f'{output_path}/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"Saved training history to {history_path}")
    
    def load_trained_models(self, model_path: str) -> None:
        """Load trained models from disk."""
        logger.info(f"Loading trained models from {model_path}")
        
        # Load clustering models
        clustering_path = f'{model_path}/clustering_models.pkl'
        if Path(clustering_path).exists():
            with open(clustering_path, 'rb') as f:
                self.trained_models['clustering'] = pickle.load(f)
            logger.info("Loaded clustering models")
        
        # Load embedding model
        embedding_path = f'{model_path}/embedding_model'
        if Path(embedding_path).exists():
            self.trained_models['embedding'] = {
                'model': SentenceTransformer(embedding_path)
            }
            logger.info("Loaded embedding model")
        
        # Load normalization models
        normalization_path = f'{model_path}/normalization_models.json'
        if Path(normalization_path).exists():
            with open(normalization_path, 'r') as f:
                self.trained_models['normalization'] = json.load(f)
            logger.info("Loaded normalization models")
        
        # Load training history
        history_path = f'{model_path}/training_history.json'
        if Path(history_path).exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            logger.info("Loaded training history")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        summary = {
            'models_trained': list(self.trained_models.keys()),
            'training_data_size': {
                'clustering_keys': len(self.training_data.get('clustering_data', {})),
                'embedding_pairs': sum(len(data['pairs']) for data in self.training_data.get('embedding_data', {}).values()),
                'normalization_examples': sum(len(data['normalized_values']) for data in self.training_data.get('normalization_data', {}).values())
            },
            'training_history': self.training_history
        }
        
        return summary