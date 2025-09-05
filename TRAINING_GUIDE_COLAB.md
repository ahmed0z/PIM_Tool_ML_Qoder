# 🎯 Complete Model Training Guide for AutoPatternChecker in Colab

## Overview

Yes! You can train and fine-tune models in AutoPatternChecker. This guide shows you how to:

1. **Train Clustering Models** - Optimize HDBSCAN and KMeans parameters
2. **Fine-tune Embedding Models** - Improve semantic similarity detection
3. **Train Normalization Models** - Learn better text cleaning rules
4. **Evaluate Model Performance** - Measure improvements

## 🚀 Step-by-Step Training in Colab

### Step 1: Setup and Install Dependencies

```python
# Install additional training dependencies
!pip install -q torch torchvision torchaudio
!pip install -q scikit-optimize optuna
!pip install -q plotly matplotlib seaborn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import AutoPatternChecker modules
import sys
sys.path.append('/content/autopatternchecker')

from autopatternchecker import *
from autopatternchecker.training import ModelTrainer
from autopatternchecker.utils import setup_logging

# Setup logging
setup_logging('INFO')
print("✅ Training environment ready!")
```

### Step 2: Load Your Data and Run Initial Pipeline

```python
# Configuration for training
config = {
    'key_columns': ['key_part1', 'key_part2', 'key_part3'],
    'value_column': 'value',
    'chunk_size': 10000,
    'encoding': 'utf-8',
    'min_signature_frequency': 5,
    'rare_signature_pct_threshold': 1.0,
    'tfidf_ngram_range': [2, 4],
    'tfidf_max_features': 2000,
    'hdbscan_min_cluster_size': 5,
    'hdbscan_min_samples': 2,
    'kmeans_n_clusters': 8,
    'embedding_model': 'all-MiniLM-L6-v2',
    'embedding_batch_size': 64,
    'use_embeddings': True,
    'index_type': 'flat',
    'metric': 'cosine',
    'review_batch_size': 100,
    'auto_accept_confidence_threshold': 0.95,
    'manual_review_threshold_lower': 0.6,
    'manual_review_threshold_upper': 0.95,
    'retrain_threshold': 10
}

# Paths
BASE_PATH = '/content/drive/MyDrive/autopatternchecker'
DATA_PATH = f'{BASE_PATH}/data'
OUTPUT_PATH = f'{BASE_PATH}/output'
INDICES_PATH = f'{BASE_PATH}/indices'
MODELS_PATH = f'{BASE_PATH}/trained_models'

# Create directories
Path(BASE_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(INDICES_PATH).mkdir(parents=True, exist_ok=True)
Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

print(f"✅ Directories created. Models will be saved to: {MODELS_PATH}")
```

### Step 3: Upload and Process Your Data

```python
# Upload your CSV file
from google.colab import files
uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]

# Move to data directory
import shutil
shutil.move(csv_filename, f'{DATA_PATH}/{csv_filename}')
csv_path = f'{DATA_PATH}/{csv_filename}'

print(f"✅ Data uploaded: {csv_filename}")

# Process the data
print("🔄 Processing data...")
ingester = DataIngester(config)
processed_df, key_stats_df = ingester.process_file(csv_path)

print(f"✅ Processed {len(processed_df)} rows")
print(f"✅ Found {len(key_stats_df)} unique composite keys")
```

### Step 4: Initialize Model Trainer

```python
# Initialize model trainer
trainer = ModelTrainer(config)

# Prepare training data
print("🔄 Preparing training data...")
training_data = trainer.prepare_training_data(key_profiles, processed_df)

print(f"✅ Training data prepared:")
print(f"  - Clustering data: {len(training_data['clustering_data'])} keys")
print(f"  - Embedding data: {len(training_data['embedding_data'])} keys")
print(f"  - Normalization data: {len(training_data['normalization_data'])} keys")
```

### Step 5: Train Clustering Models

```python
# Train clustering models
print("🔄 Training clustering models...")
clustering_models = trainer.train_clustering_models(training_data)

print(f"✅ Clustering models trained for {len(clustering_models)} keys")

# Show training results
for key, model_data in list(clustering_models.items())[:3]:
    print(f"\nKey: {key}")
    print(f"  Best model: {model_data['model']['type']}")
    print(f"  Score: {model_data['model']['score']:.3f}")
    print(f"  Parameters: {model_data['model']['params']}")
```

### Step 6: Train Embedding Models

```python
# Train embedding models
print("🔄 Training embedding models...")
embedding_models = trainer.train_embedding_models(training_data)

if embedding_models:
    print(f"✅ Embedding model trained")
    print(f"  Base model: {embedding_models['base_model']}")
    print(f"  Training examples: {embedding_models['training_examples']}")
else:
    print("⚠️  No embedding training data available")
```

### Step 7: Train Normalization Models

```python
# Train normalization models
print("🔄 Training normalization models...")
normalization_models = trainer.train_normalization_models(training_data)

if normalization_models:
    print(f"✅ Normalization model trained")
    print(f"  Rules learned: {len(normalization_models['rules'])}")
    print(f"  Training examples: {normalization_models['training_examples']}")
    
    # Show learned rules
    for rule in normalization_models['rules']:
        print(f"    - {rule['name']}: {rule['confidence']:.2f} confidence")
else:
    print("⚠️  No normalization training data available")
```

### Step 8: Evaluate Models

```python
# Create test data (split from training data)
from sklearn.model_selection import train_test_split

# Split data for evaluation
test_data = {}
for key, data in training_data['clustering_data'].items():
    if len(data['features']) > 20:  # Only keys with enough data
        train_features, test_features, train_labels, test_labels = train_test_split(
            data['features'], data['labels'], test_size=0.3, random_state=42
        )
        test_data['clustering_data'] = test_data.get('clustering_data', {})
        test_data['clustering_data'][key] = {
            'features': test_features,
            'labels': test_labels
        }

# Evaluate models
print("🔄 Evaluating models...")
evaluation_results = trainer.evaluate_models(test_data)

print("✅ Model evaluation completed:")
print(json.dumps(evaluation_results, indent=2, default=str))
```

### Step 9: Visualize Training Results

```python
# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Clustering performance
if 'clustering' in evaluation_results:
    scores = [data['silhouette_score'] for data in evaluation_results['clustering'].values()]
    axes[0, 0].hist(scores, bins=10, alpha=0.7)
    axes[0, 0].set_title('Clustering Silhouette Scores')
    axes[0, 0].set_xlabel('Silhouette Score')
    axes[0, 0].set_ylabel('Frequency')

# 2. Model types used
if clustering_models:
    model_types = [data['model']['type'] for data in clustering_models.values()]
    model_counts = pd.Series(model_types).value_counts()
    axes[0, 1].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Clustering Model Types')

# 3. Training data size
data_sizes = trainer.get_training_summary()['training_data_size']
axes[1, 0].bar(data_sizes.keys(), data_sizes.values())
axes[1, 0].set_title('Training Data Sizes')
axes[1, 0].set_ylabel('Count')

# 4. Model performance comparison
if 'clustering' in evaluation_results:
    keys = list(evaluation_results['clustering'].keys())[:10]
    hdbscan_scores = [evaluation_results['clustering'][k].get('silhouette_score', 0) for k in keys]
    axes[1, 1].bar(range(len(keys)), hdbscan_scores)
    axes[1, 1].set_title('Top 10 Keys - Clustering Performance')
    axes[1, 1].set_xlabel('Key Index')
    axes[1, 1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
```

### Step 10: Save Trained Models

```python
# Save all trained models
print("🔄 Saving trained models...")
trainer.save_trained_models(MODELS_PATH)

print(f"✅ Models saved to {MODELS_PATH}")

# Show training summary
summary = trainer.get_training_summary()
print("\n📊 Training Summary:")
print(f"  Models trained: {summary['models_trained']}")
print(f"  Clustering keys: {summary['training_data_size']['clustering_keys']}")
print(f"  Embedding pairs: {summary['training_data_size']['embedding_pairs']}")
print(f"  Normalization examples: {summary['training_data_size']['normalization_examples']}")
```

### Step 11: Test Trained Models

```python
# Test the trained models
print("🔄 Testing trained models...")

# Test clustering
if 'clustering' in trainer.trained_models:
    test_key = list(trainer.trained_models['clustering'].keys())[0]
    test_model = trainer.trained_models['clustering'][test_key]['model']['model']
    test_features = trainer.trained_models['clustering'][test_key]['features']
    
    # Predict clusters
    predictions = test_model.fit_predict(test_features)
    print(f"✅ Clustering test - Key: {test_key}")
    print(f"  Predicted clusters: {len(set(predictions))}")
    print(f"  Sample predictions: {predictions[:5]}")

# Test embedding
if 'embedding' in trainer.trained_models:
    test_model = trainer.trained_models['embedding']['model']
    test_texts = ["SMB Mini Jack", "SC Plug", "Resistor 1kΩ"]
    
    # Generate embeddings
    embeddings = test_model.encode(test_texts)
    print(f"✅ Embedding test - Generated {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

# Test normalization
if 'normalization' in trainer.trained_models:
    rules = trainer.trained_models['normalization']['rules']
    print(f"✅ Normalization test - Learned {len(rules)} rules")
    for rule in rules:
        print(f"  - {rule['name']}: {rule['confidence']:.2f} confidence")
```

### Step 12: Create Training Report

```python
# Create comprehensive training report
def create_training_report():
    report = {
        'training_date': datetime.now().isoformat(),
        'config': config,
        'data_info': {
            'total_rows': len(processed_df),
            'unique_keys': len(key_stats_df),
            'training_keys': len(training_data['clustering_data'])
        },
        'models_trained': trainer.get_training_summary(),
        'evaluation_results': evaluation_results,
        'performance_metrics': {}
    }
    
    # Calculate performance metrics
    if 'clustering' in evaluation_results:
        scores = [data['silhouette_score'] for data in evaluation_results['clustering'].values()]
        report['performance_metrics']['clustering'] = {
            'mean_silhouette_score': np.mean(scores),
            'std_silhouette_score': np.std(scores),
            'min_silhouette_score': np.min(scores),
            'max_silhouette_score': np.max(scores)
        }
    
    return report

# Generate and save report
training_report = create_training_report()
report_path = f'{OUTPUT_PATH}/training_report.json'

with open(report_path, 'w') as f:
    json.dump(training_report, f, indent=2, default=str)

print(f"✅ Training report saved to {report_path}")

# Display key metrics
print("\n📈 Key Performance Metrics:")
if 'clustering' in training_report['performance_metrics']:
    metrics = training_report['performance_metrics']['clustering']
    print(f"  Mean Silhouette Score: {metrics['mean_silhouette_score']:.3f}")
    print(f"  Score Range: {metrics['min_silhouette_score']:.3f} - {metrics['max_silhouette_score']:.3f}")
```

### Step 13: Use Trained Models in Production

```python
# Load trained models for production use
print("🔄 Loading trained models for production...")

# Create new trainer instance
production_trainer = ModelTrainer(config)
production_trainer.load_trained_models(MODELS_PATH)

print("✅ Trained models loaded for production use")

# Test production models
if 'clustering' in production_trainer.trained_models:
    print(f"  Clustering models: {len(production_trainer.trained_models['clustering'])} keys")
if 'embedding' in production_trainer.trained_models:
    print(f"  Embedding model: {production_trainer.trained_models['embedding']['base_model']}")
if 'normalization' in production_trainer.trained_models:
    print(f"  Normalization rules: {len(production_trainer.trained_models['normalization']['rules'])}")
```

## 🎯 Advanced Training Options

### Hyperparameter Optimization

```python
# Advanced hyperparameter optimization using Optuna
!pip install -q optuna

import optuna

def optimize_clustering_hyperparameters(trial, features, labels):
    """Optimize clustering hyperparameters using Optuna."""
    
    # HDBSCAN parameters
    min_cluster_size = trial.suggest_int('min_cluster_size', 3, 20)
    min_samples = trial.suggest_int('min_samples', 1, 10)
    metric = trial.suggest_categorical('metric', ['euclidean', 'cosine', 'manhattan'])
    
    # KMeans parameters
    n_clusters = trial.suggest_int('n_clusters', 2, min(20, len(features) // 2))
    
    # Train both models
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Evaluate both
    hdbscan_labels = hdbscan_model.fit_predict(features)
    kmeans_labels = kmeans_model.fit_predict(features)
    
    # Calculate scores
    hdbscan_score = silhouette_score(features, hdbscan_labels) if len(set(hdbscan_labels)) > 1 else -1
    kmeans_score = silhouette_score(features, kmeans_labels) if len(set(kmeans_labels)) > 1 else -1
    
    # Return best score
    return max(hdbscan_score, kmeans_score)

# Run optimization for a specific key
if training_data['clustering_data']:
    key = list(training_data['clustering_data'].keys())[0]
    features = training_data['clustering_data'][key]['features']
    labels = training_data['clustering_data'][key]['labels']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_clustering_hyperparameters(trial, features, labels), n_trials=50)
    
    print(f"✅ Best parameters: {study.best_params}")
    print(f"✅ Best score: {study.best_value:.3f}")
```

### Custom Loss Functions

```python
# Custom loss function for embedding training
import torch
import torch.nn as nn

class CustomContrastiveLoss(nn.Module):
    """Custom contrastive loss for better similarity learning."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # Implement custom loss logic
        # This is a simplified example
        distances = torch.cdist(embeddings, embeddings)
        
        # Positive pairs (same label)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Negative pairs (different labels)
        negative_mask = ~positive_mask
        
        # Contrastive loss
        positive_loss = torch.sum(positive_mask.float() * distances**2)
        negative_loss = torch.sum(negative_mask.float() * torch.clamp(self.margin - distances, min=0)**2)
        
        return positive_loss + negative_loss
```

## 📊 Training Monitoring

### Real-time Training Progress

```python
# Monitor training progress
import time
from IPython.display import clear_output

def monitor_training_progress():
    """Monitor training progress in real-time."""
    
    start_time = time.time()
    
    for epoch in range(10):  # Example training loop
        # Simulate training
        time.sleep(1)
        
        # Clear output and show progress
        clear_output(wait=True)
        
        elapsed = time.time() - start_time
        progress = (epoch + 1) / 10 * 100
        
        print(f"🔄 Training Progress: {progress:.1f}%")
        print(f"⏱️  Elapsed Time: {elapsed:.1f}s")
        print(f"📊 Epoch: {epoch + 1}/10")
        
        # Show metrics (example)
        print(f"📈 Loss: {0.5 * (1 - epoch/10):.3f}")
        print(f"📈 Accuracy: {0.5 + epoch/20:.3f}")

# Run monitoring
monitor_training_progress()
```

## 🎉 Training Complete!

After running all these steps, you'll have:

- ✅ **Trained Clustering Models** - Optimized for your specific data
- ✅ **Fine-tuned Embedding Models** - Better semantic similarity
- ✅ **Learned Normalization Rules** - Improved text cleaning
- ✅ **Performance Metrics** - Quantified improvements
- ✅ **Production-Ready Models** - Ready for deployment

## 🚀 Next Steps

1. **Deploy Trained Models** - Use in your production API
2. **Monitor Performance** - Track model performance over time
3. **Retrain Periodically** - Update models with new data
4. **A/B Testing** - Compare trained vs untrained models
5. **Hyperparameter Tuning** - Further optimize parameters

Your AutoPatternChecker system is now trained and optimized for your specific data! 🎯