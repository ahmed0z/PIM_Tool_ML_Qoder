# 🚀 Alternative Ways to Run AutoPatternChecker (Persistent Solutions)

Since Google Colab deletes files after runtime ends, here are better alternatives that persist your work:

## Option 1: Google Colab Pro + Google Drive (Recommended)

### Setup Persistent Environment
```python
# In Colab, run this first to set up persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create persistent directories
import os
os.makedirs('/content/drive/MyDrive/autopatternchecker', exist_ok=True)
os.makedirs('/content/drive/MyDrive/autopatternchecker/data', exist_ok=True)
os.makedirs('/content/drive/MyDrive/autopatternchecker/output', exist_ok=True)
os.makedirs('/content/drive/MyDrive/autopatternchecker/indices', exist_ok=True)

# Clone or upload your code to Google Drive
!cd /content/drive/MyDrive/autopatternchecker && git clone https://github.com/yourusername/autopatternchecker.git .

# Set working directory
%cd /content/drive/MyDrive/autopatternchecker
```

### Run with Persistent Storage
```python
# All your work will be saved to Google Drive
BASE_PATH = '/content/drive/MyDrive/autopatternchecker'
# ... rest of your code
```

**Pros**: Free, persistent, easy to use
**Cons**: Limited compute time, files can be lost if not saved properly

## Option 2: Kaggle Notebooks (Free Alternative)

### Setup
1. Go to [Kaggle.com](https://www.kaggle.com)
2. Create account and go to "Notebooks"
3. Create new notebook
4. Upload your project files or clone from GitHub

### Run in Kaggle
```python
# Install dependencies
!pip install pandas numpy scikit-learn hdbscan faiss-cpu sentence-transformers fastapi uvicorn pyyaml

# Clone your repository
!git clone https://github.com/yourusername/autopatternchecker.git
%cd autopatternchecker

# Your code here...
```

**Pros**: Free, persistent, good GPU access
**Cons**: Limited to 9 hours per session, internet required

## Option 3: Local Development (Best for Development)

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- 10GB+ free disk space

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/autopatternchecker.git
cd autopatternchecker

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the pipeline
python -m autopatternchecker.pipeline
```

### Jupyter Notebook Setup
```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Open colab_notebook_full_pipeline.ipynb
```

**Pros**: Full control, persistent, fast
**Cons**: Requires local setup, may need powerful hardware

## Option 4: Google Cloud Platform (Production Ready)

### Setup GCP Environment
```bash
# Create VM instance
gcloud compute instances create autopatternchecker \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# SSH into instance
gcloud compute ssh autopatternchecker --zone=us-central1-a
```

### Install on GCP
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install system dependencies
sudo apt install git gcc g++ curl

# Clone repository
git clone https://github.com/yourusername/autopatternchecker.git
cd autopatternchecker

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the service
uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000
```

**Pros**: Scalable, persistent, professional
**Cons**: Costs money, requires GCP knowledge

## Option 5: Docker Desktop (Local Container)

### Setup Docker
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Clone your repository
3. Build and run container

### Run with Docker
```bash
# Build the image
docker build -t autopatternchecker:latest .

# Run the container
docker run -d \
  --name autopatternchecker \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/indices:/app/indices \
  autopatternchecker:latest

# Check if running
docker ps
```

**Pros**: Isolated environment, easy deployment
**Cons**: Requires Docker knowledge

## Option 6: Replit (Online IDE)

### Setup
1. Go to [Replit.com](https://replit.com)
2. Create new Python repl
3. Upload your project files

### Run in Replit
```python
# Install dependencies
!pip install pandas numpy scikit-learn hdbscan faiss-cpu sentence-transformers fastapi uvicorn pyyaml

# Your code here...
```

**Pros**: Free tier available, persistent, easy to share
**Cons**: Limited resources on free tier

## Option 7: GitHub Codespaces (Microsoft)

### Setup
1. Push your code to GitHub
2. Go to your repository
3. Click "Code" → "Codespaces" → "Create codespace"

### Run in Codespaces
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the service
uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000
```

**Pros**: Integrated with GitHub, persistent, good resources
**Cons**: Limited free hours per month

## Option 8: AWS SageMaker (ML Platform)

### Setup
1. Create AWS account
2. Go to SageMaker → Notebook Instances
3. Create new instance
4. Upload your notebook

### Run in SageMaker
```python
# Install dependencies
!pip install pandas numpy scikit-learn hdbscan faiss-cpu sentence-transformers fastapi uvicorn pyyaml

# Your code here...
```

**Pros**: ML-optimized, persistent, scalable
**Cons**: Costs money, AWS complexity

## 🎯 Recommended Approach

### For Learning/Experimentation:
1. **Google Colab Pro + Google Drive** - Best free option
2. **Kaggle Notebooks** - Good alternative
3. **Local Jupyter** - If you have good hardware

### For Development:
1. **Local Development** - Best for active development
2. **Docker Desktop** - Good for testing deployments
3. **GitHub Codespaces** - Good for collaboration

### For Production:
1. **Google Cloud Platform** - Scalable and reliable
2. **AWS SageMaker** - ML-optimized
3. **Docker + Cloud Provider** - Most flexible

## 🔧 Making Colab More Persistent

If you must use Colab, here are tips to make it more persistent:

### 1. Save Everything to Google Drive
```python
# Always save to Google Drive
BASE_PATH = '/content/drive/MyDrive/autopatternchecker'
```

### 2. Create Checkpoint System
```python
# Save progress at each step
import pickle

def save_checkpoint(data, step):
    with open(f'{BASE_PATH}/checkpoint_{step}.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(step):
    with open(f'{BASE_PATH}/checkpoint_{step}.pkl', 'rb') as f:
        return pickle.load(f)

# Use checkpoints
if os.path.exists(f'{BASE_PATH}/checkpoint_1.pkl'):
    processed_df, key_stats_df = load_checkpoint(1)
else:
    processed_df, key_stats_df = ingester.process_file(csv_path)
    save_checkpoint((processed_df, key_stats_df), 1)
```

### 3. Use Colab Pro Features
- **Colab Pro**: $10/month, longer runtimes, better GPUs
- **Colab Pro+**: $50/month, even longer runtimes, priority access

### 4. Backup Strategy
```python
# Create backup of all artifacts
import shutil
import datetime

def create_backup():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'{BASE_PATH}/backups/backup_{timestamp}'
    shutil.copytree(f'{BASE_PATH}/output', f'{backup_path}/output')
    shutil.copytree(f'{BASE_PATH}/indices', f'{backup_path}/indices')
    print(f"Backup created: {backup_path}")

# Run backup before ending session
create_backup()
```

## 🚀 Quick Start Commands

### Local Development (Recommended)
```bash
# Clone and setup
git clone https://github.com/yourusername/autopatternchecker.git
cd autopatternchecker
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .

# Run pipeline
python -c "
from autopatternchecker import *
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Process data
config = {'key_columns': ['col1', 'col2', 'col3'], 'value_column': 'value'}
ingester = DataIngester(config)
processed_df, key_stats_df = ingester.process_file('your_data.csv')

# Generate profiles
profiler = PatternProfiler(config)
key_profiles = profiler.analyze_key_patterns(key_stats_df, processed_df)

print('Pipeline completed!')
"

# Start API
uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000
```

### Docker (Easy Deployment)
```bash
# Build and run
docker build -t autopatternchecker .
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data autopatternchecker

# Test API
curl http://localhost:8000/health
```

Choose the option that best fits your needs and budget!