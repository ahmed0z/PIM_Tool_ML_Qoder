# 🚀 Complete Step-by-Step Guide: Clone and Run AutoPatternChecker in Colab

## Step 1: Clone the Repository

### Option A: Clone from GitHub (if you've pushed to GitHub)
```bash
# In your local terminal
git clone https://github.com/yourusername/autopatternchecker.git
cd autopatternchecker
```

### Option B: Download the Project Files
If you haven't pushed to GitHub yet, you can download the project files directly from your workspace.

## Step 2: Upload to Google Colab

### Method 1: Upload Individual Files
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "New Notebook"
3. Click the folder icon (📁) in the left sidebar
4. Click "Upload to session storage"
5. Upload these files:
   - `colab_notebook_full_pipeline.ipynb`
   - All files from the `autopatternchecker/` folder
   - `requirements.txt`
   - `configs/` folder

### Method 2: Upload as ZIP
1. Create a ZIP file of your project
2. In Colab, click "Files" → "Upload to session storage"
3. Upload the ZIP file
4. Extract it:
```python
!unzip your_project.zip
```

### Method 3: Clone from GitHub in Colab
```python
# Run this in a Colab cell
!git clone https://github.com/yourusername/autopatternchecker.git
!cd autopatternchecker
```

## Step 3: Set Up the Environment

### Cell 1: Install Dependencies
```python
# Install all required packages
!pip install -q pandas numpy scikit-learn hdbscan faiss-cpu sentence-transformers fastapi uvicorn pyyaml python-multipart pydantic

# Mount Google Drive (optional, for saving artifacts)
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Import Libraries and Setup
```python
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
sys.path.append('/content/autopatternchecker')  # Adjust path if needed

from autopatternchecker import (
    DataIngester, PatternProfiler, NormalizationEngine,
    ClusterAnalyzer, EmbeddingGenerator, FAISSIndexer,
    ActiveLearningManager, create_app
)
from autopatternchecker.utils import setup_logging

# Setup logging
setup_logging('INFO')
print("✅ Setup complete!")
```

## Step 4: Configure the System

### Cell 3: Configuration
```python
# Configuration - ADJUST THESE FOR YOUR DATA
config = {
    'key_columns': ['key_part1', 'key_part2', 'key_part3'],  # Change to your CSV column names
    'value_column': 'value',  # Change to your value column name
    'chunk_size': 10000,
    'encoding': 'utf-8',
    'min_signature_frequency': 5,
    'rare_signature_pct_threshold': 1.0,
    'tfidf_ngram_range': [2, 4],
    'tfidf_max_features': 2000,
    'hdbscan_min_cluster_size': 5,
    'hdbscan_min_samples': 2,
    'kmeans_n_clusters': 8,
    'embedding_model': 'all-MiniLM-L6-v2',  # Use this for speed
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

# Paths - Choose one option
# Option 1: Use Google Drive (recommended for saving artifacts)
BASE_PATH = '/content/drive/MyDrive/autopatternchecker'

# Option 2: Use local storage (temporary, will be lost when session ends)
# BASE_PATH = '/content/autopatternchecker'

DATA_PATH = f'{BASE_PATH}/data'
OUTPUT_PATH = f'{BASE_PATH}/output'
INDICES_PATH = f'{BASE_PATH}/indices'

# Create directories
Path(BASE_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(INDICES_PATH).mkdir(parents=True, exist_ok=True)

print(f"✅ Configuration set up. Output will be saved to: {OUTPUT_PATH}")
```

## Step 5: Upload Your Data

### Cell 4: Upload CSV File
```python
# Upload your CSV file
from google.colab import files
uploaded = files.upload()

# Get the uploaded filename
csv_filename = list(uploaded.keys())[0]
print(f"Uploaded file: {csv_filename}")

# Move to data directory
import shutil
shutil.move(csv_filename, f'{DATA_PATH}/{csv_filename}')
csv_path = f'{DATA_PATH}/{csv_filename}'
print(f"File moved to: {csv_path}")
```

### Cell 5: Preview Your Data
```python
# Preview the data to make sure it's correct
df_preview = pd.read_csv(csv_path, nrows=10)
print("Data preview:")
print(df_preview.head())
print(f"\nColumns: {list(df_preview.columns)}")
print(f"Data types:\n{df_preview.dtypes}")

# Check if your column names match the config
print(f"\nConfig expects columns: {config['key_columns'] + [config['value_column']]}")
print(f"Your CSV has columns: {list(df_preview.columns)}")

# If they don't match, update the config:
# config['key_columns'] = ['your_col1', 'your_col2', 'your_col3']
# config['value_column'] = 'your_value_column'
```

## Step 6: Run the Pipeline

### Cell 6: Data Ingestion
```python
# Initialize data ingester
ingester = DataIngester(config)

# Process the CSV file
print("🔄 Processing CSV file...")
processed_df, key_stats_df = ingester.process_file(csv_path)

print(f"✅ Processed {len(processed_df)} rows")
print(f"✅ Found {len(key_stats_df)} unique composite keys")
print(f"\nTop 5 keys by count:")
print(key_stats_df.nlargest(5, 'count')[['composite_key', 'count', 'unique_values', 'unique_ratio']])
```

### Cell 7: Pattern Profiling
```python
# Initialize pattern profiler
profiler = PatternProfiler(config)

# Generate key profiles
print("🔄 Generating pattern profiles...")
key_profiles = profiler.analyze_key_patterns(key_stats_df, processed_df)

print(f"✅ Generated profiles for {len(key_profiles)} keys")

# Show example profile
example_key = list(key_profiles.keys())[0]
example_profile = key_profiles[example_key]
print(f"\nExample profile for key: {example_key}")
print(f"Count: {example_profile['count']}")
print(f"Unique values: {example_profile['unique_values']}")
print(f"Is free text: {example_profile['is_free_text']}")
print(f"Is numeric unit: {example_profile['is_numeric_unit']}")
print(f"\nTop signatures:")
for sig in example_profile['top_signatures'][:3]:
    print(f"  {sig['sig']}: {sig['count']} ({sig['pct']}%)")
print(f"\nCandidate regex: {example_profile['candidate_regex']}")
```

### Cell 8: Normalization Rules
```python
# Initialize normalization engine
normalizer = NormalizationEngine(config)

# Generate normalization rules
print("🔄 Generating normalization rules...")
normalization_rules = normalizer.generate_normalization_rules(key_profiles)

print(f"✅ Generated normalization rules for {len(normalization_rules)} keys")

# Show example rules
example_key = list(normalization_rules.keys())[0]
example_rules = normalization_rules[example_key]
print(f"\nExample rules for key: {example_key}")
print(f"Confidence: {example_rules['confidence']:.2f}")
print(f"Rules:")
for rule in example_rules['rules']:
    print(f"  - {rule['name']} (priority: {rule['priority']})")

# Load rules into the engine
normalizer.load_normalization_rules(normalization_rules)
print("\n✅ Normalization rules loaded")
```

### Cell 9: Clustering Analysis
```python
# Initialize cluster analyzer
clusterer = ClusterAnalyzer(config)

# Prepare data for clustering (sample top keys)
top_keys = key_stats_df.nlargest(10, 'count')['composite_key'].tolist()
key_values = {}

for key in top_keys:
    values = processed_df[processed_df['__composite_key__'] == key]['value'].dropna().tolist()
    if len(values) >= 10:  # Only cluster keys with enough data
        key_values[key] = values

print(f"🔄 Clustering {len(key_values)} keys...")

# Perform clustering
clustering_results = clusterer.cluster_all_keys(key_profiles, key_values)

print(f"✅ Completed clustering for {len(clustering_results)} keys")

# Show clustering results
for key, result in list(clustering_results.items())[:3]:
    print(f"\nKey: {key}")
    print(f"  Clusters: {result['n_clusters']}")
    print(f"  Method: {result['method']}")
    print(f"  Silhouette score: {result['silhouette_score']:.3f}")
    if result['clusters']:
        print(f"  Top cluster: {result['clusters'][0]['pattern_signature']} ({result['clusters'][0]['size']} samples)")
```

### Cell 10: Embedding Generation
```python
# Initialize embedding generator
embedder = EmbeddingGenerator(config)

# Generate embeddings for all keys
print("🔄 Generating embeddings...")
embeddings_data = embedder.generate_embeddings_for_all_keys(key_values)

print(f"✅ Generated embeddings for {len(embeddings_data)} keys")

# Show embedding statistics
total_embeddings = sum(data['count'] for data in embeddings_data.values())
print(f"Total embeddings generated: {total_embeddings}")

# Show example embedding info
example_key = list(embeddings_data.keys())[0]
example_data = embeddings_data[example_key]
print(f"\nExample embedding info for key: {example_key}")
print(f"  Count: {example_data['count']}")
print(f"  Dimension: {example_data['dimension']}")
print(f"  Model: {example_data['model_name']}")
```

### Cell 11: FAISS Index Building
```python
# Initialize FAISS indexer
indexer = FAISSIndexer(config)

# Build indices for all keys
print("🔄 Building FAISS indices...")
index_results = indexer.build_all_indices(embeddings_data)

print(f"✅ Built indices for {len(index_results)} keys")

# Save indices to disk
indexer.save_indices(INDICES_PATH)
print(f"✅ Indices saved to {INDICES_PATH}")

# Show index info
index_info = indexer.get_index_info()
print(f"\nIndex configuration:")
print(f"  Type: {index_info['index_type']}")
print(f"  Metric: {index_info['metric']}")
print(f"  Total indices: {index_info['total_indices']}")
```

### Cell 12: Save All Artifacts
```python
# Save all artifacts
print("🔄 Saving artifacts...")

# Save key profiles
profiles_path = f'{OUTPUT_PATH}/key_profiles.json'
with open(profiles_path, 'w') as f:
    json.dump(key_profiles, f, indent=2, default=str)
print(f"✅ Key profiles saved to {profiles_path}")

# Save normalization rules
rules_path = f'{OUTPUT_PATH}/normalization_rules.json'
with open(rules_path, 'w') as f:
    json.dump(normalization_rules, f, indent=2, default=str)
print(f"✅ Normalization rules saved to {rules_path}")

# Save clustering results
clustering_path = f'{OUTPUT_PATH}/clustering_results.json'
with open(clustering_path, 'w') as f:
    json.dump(clustering_results, f, indent=2, default=str)
print(f"✅ Clustering results saved to {clustering_path}")

# Save embeddings data
embeddings_path = f'{OUTPUT_PATH}/embeddings_data.pkl'
embedder.save_embeddings(embeddings_data, embeddings_path)
print(f"✅ Embeddings data saved to {embeddings_path}")

# Save configuration
config_path = f'{OUTPUT_PATH}/config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"✅ Configuration saved to {config_path}")

print("\n🎉 All artifacts saved successfully!")
```

## Step 7: Start the API

### Cell 13: Start FastAPI Service
```python
# Update config with file paths
api_config = config.copy()
api_config.update({
    'key_profiles_path': profiles_path,
    'normalization_rules_path': rules_path,
    'faiss_indices_path': INDICES_PATH
})

# Create FastAPI app
app = create_app(api_config)

print("✅ FastAPI app created")
print("\nTo start the API server, run the next cell")
```

### Cell 14: Run the API Server
```python
# Start the API server
import uvicorn
from threading import Thread

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Start server in background
server_thread = Thread(target=run_server, daemon=True)
server_thread.start()

print("🚀 API server started on http://localhost:8000")
print("\nAPI Documentation: http://localhost:8000/docs")
print("Health check: http://localhost:8000/health")

# Wait a moment for server to start
import time
time.sleep(3)

# Test the API
import requests
try:
    response = requests.get("http://localhost:8000/health")
    print(f"\n✅ API is running! Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"❌ API test failed: {e}")
```

## Step 8: Test the API

### Cell 15: Test Validation
```python
# Test validation with sample data
import requests

# Get a sample composite key
sample_key = list(key_profiles.keys())[0]
key_parts = sample_key.split('||')
sample_value = processed_df[processed_df['__composite_key__'] == sample_key]['value'].iloc[0]

print(f"Testing validation for key: {sample_key}")
print(f"Sample value: {sample_value}")

# Test validation request
validation_request = {
    "key_parts": key_parts,
    "value": sample_value,
    "metadata": {"test": True}
}

try:
    response = requests.post("http://localhost:8000/validate", json=validation_request)
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Validation successful!")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Issues: {result['issues']}")
        print(f"Rules applied: {result['rules_applied']}")
        print(f"Processing time: {result['processing_time_ms']}ms")
    else:
        print(f"❌ Validation failed: {response.status_code}")
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Request failed: {e}")
```

### Cell 16: Test Multiple Values
```python
# Test with multiple values
test_values = [
    sample_value,  # Original value
    sample_value.upper(),  # Uppercase version
    sample_value + " (modified)",  # Modified version
    "completely different value",  # Different value
    ""  # Empty value
]

print("Testing multiple values:")
print("=" * 50)

for i, test_value in enumerate(test_values):
    print(f"\nTest {i+1}: '{test_value}'")
    
    validation_request = {
        "key_parts": key_parts,
        "value": test_value
    }
    
    try:
        response = requests.post("http://localhost:8000/validate", json=validation_request)
        if response.status_code == 200:
            result = response.json()
            print(f"  Verdict: {result['verdict']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Issues: {result['issues']}")
            if result['suggested_fix']:
                print(f"  Suggested fix: {result['suggested_fix']}")
        else:
            print(f"  ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
```

## Step 9: Access the API Documentation

### Cell 17: Open API Documentation
```python
# The API documentation is available at:
print("🌐 API Documentation: http://localhost:8000/docs")
print("🔍 Interactive API Explorer: http://localhost:8000/redoc")

# You can also access it directly in Colab
from IPython.display import IFrame
IFrame(src="http://localhost:8000/docs", width=800, height=600)
```

## 🎉 You're Done!

Your AutoPatternChecker system is now running in Colab with:

- ✅ **Data Processed**: Your CSV data has been analyzed
- ✅ **Patterns Learned**: Signatures and normalization rules generated
- ✅ **API Running**: Validation service available at `http://localhost:8000`
- ✅ **Artifacts Saved**: All results saved to Google Drive (if mounted)

## 🔧 Troubleshooting

### If you get import errors:
```python
# Make sure the path is correct
import sys
sys.path.append('/content/autopatternchecker')  # Adjust this path
```

### If you get memory errors:
```python
# Reduce batch sizes in config
config['chunk_size'] = 5000
config['embedding_batch_size'] = 32
```

### If the API doesn't start:
```python
# Check if port 8000 is available
!lsof -i :8000
```

### If you want to stop the API:
```python
# The API will stop when you restart the runtime or close the notebook
```

## 📁 Files Created

All these files will be saved to your Google Drive:
- `key_profiles.json` - Pattern profiles
- `normalization_rules.json` - Cleaning rules
- `clustering_results.json` - Clustering analysis
- `embeddings_data.pkl` - Generated embeddings
- `indices/` - FAISS search indices
- `config.yaml` - Configuration used

You can now use the API to validate new values or download the artifacts for production use!