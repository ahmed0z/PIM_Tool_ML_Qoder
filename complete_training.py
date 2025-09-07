#!/usr/bin/env python3
"""
Complete the training by generating embeddings and building indices
for the already processed key profiles and normalization rules.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add the project root to Python path
sys.path.insert(0, '.')

from autopatternchecker.embeddings import EmbeddingGenerator
from autopatternchecker.indexing import FAISSIndexer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

def load_existing_data(output_dir: str) -> tuple:
    """Load existing key profiles and normalization rules."""
    profiles_file = os.path.join(output_dir, 'key_profiles.json')
    rules_file = os.path.join(output_dir, 'normalization_rules.json')
    
    if not os.path.exists(profiles_file):
        raise FileNotFoundError(f"Key profiles not found: {profiles_file}")
    if not os.path.exists(rules_file):
        raise FileNotFoundError(f"Normalization rules not found: {rules_file}")
    
    print(f"📁 Loading key profiles from {profiles_file}")
    with open(profiles_file, 'r') as f:
        key_profiles = json.load(f)
    
    print(f"📁 Loading normalization rules from {rules_file}")
    with open(rules_file, 'r') as f:
        normalization_rules = json.load(f)
    
    return key_profiles, normalization_rules

def extract_all_values(key_profiles: Dict) -> List[str]:
    """Extract all unique values from key profiles for embedding."""
    all_values = set()

    for key, profile in key_profiles.items():
        if 'sample_values' in profile:
            for value in profile['sample_values']:
                if isinstance(value, str) and value.strip():
                    all_values.add(value.strip())

    return list(all_values)

def complete_training(output_dir: str = "trained_models"):
    """Complete the training by generating embeddings and building indices."""
    setup_logging()
    
    print("🚀 Completing AutoPatternChecker Training")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load existing data
        key_profiles, normalization_rules = load_existing_data(output_dir)
        print(f"✅ Loaded {len(key_profiles)} key profiles")
        print(f"✅ Loaded {len(normalization_rules)} normalization rule sets")
        
        # Extract all values for embedding
        print("\n🔄 Step 1: Extracting values for embedding...")
        all_values = extract_all_values(key_profiles)
        print(f"📊 Found {len(all_values)} unique values to embed")
        
        if not all_values:
            print("⚠️ No values found for embedding. Skipping embedding step.")
            return
        
        # Initialize embedding generator
        print("\n🔄 Step 2: Generating embeddings...")
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_batch_size': 32,
            'use_embeddings': True
        }
        embedding_generator = EmbeddingGenerator(config)

        # Generate embeddings in batches to avoid memory issues
        batch_size = 100
        embeddings_data = {}

        for i in range(0, len(all_values), batch_size):
            batch = all_values[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_values) + batch_size - 1)//batch_size}")

            batch_embeddings_array = embedding_generator.generate_embeddings(batch)

            # Convert numpy array to dictionary mapping
            for j, value in enumerate(batch):
                embeddings_data[value] = batch_embeddings_array[j].tolist()
        
        print(f"✅ Generated embeddings for {len(embeddings_data)} values")
        
        # Save embedding model info
        embedding_model_dir = os.path.join(output_dir, 'embedding_model')
        os.makedirs(embedding_model_dir, exist_ok=True)
        
        # Get embedding dimension from the first embedding
        embedding_dim = len(list(embeddings_data.values())[0]) if embeddings_data else 384

        embedding_info = {
            'model_name': embedding_generator.model_name,
            'embedding_dim': embedding_dim,
            'total_embeddings': len(embeddings_data),
            'values': list(embeddings_data.keys())
        }
        
        embedding_model_file = os.path.join(embedding_model_dir, 'model_info.json')
        with open(embedding_model_file, 'w') as f:
            json.dump(embedding_info, f, indent=2)
        
        print(f"💾 Saved embedding info to {embedding_model_file}")
        
        # Build FAISS indices
        print("\n🔄 Step 3: Building FAISS indices...")
        indexer_config = {
            'index_type': 'flat',
            'metric': 'euclidean'
        }
        faiss_indexer = FAISSIndexer(indexer_config)

        # Create indices directory
        faiss_indices_dir = os.path.join(output_dir, 'faiss_indices')
        os.makedirs(faiss_indices_dir, exist_ok=True)

        # Build and save indices
        # Convert embeddings_data to the expected format for FAISS indexer
        formatted_embeddings_data = {}
        for value, embedding in embeddings_data.items():
            formatted_embeddings_data[value] = {
                'embeddings': np.array([embedding]),  # Single embedding as array
                'values': [value],  # Single value as list
                'metadata': {}
            }

        faiss_indexer.build_all_indices(formatted_embeddings_data)
        faiss_indexer.save_indices(faiss_indices_dir)
        
        print(f"✅ Built and saved FAISS indices to {faiss_indices_dir}")
        
        # Generate final training report
        print("\n🔄 Step 4: Generating training report...")
        
        training_report = {
            "training_date": "2025-09-06T22:45:00.000000",
            "data_file": "Preset_251.csv",
            "output_directory": output_dir,
            "completion_status": "completed_embeddings_and_indexing",
            "data_stats": {
                "total_keys": len(key_profiles),
                "total_values_embedded": len(embeddings_data)
            },
            "model_stats": {
                "key_profiles": len(key_profiles),
                "normalization_rules": len(normalization_rules),
                "embedding_vectors": len(embeddings_data),
                "faiss_indices": "created"
            },
            "files_created": [
                "key_profiles.json",
                "normalization_rules.json", 
                "clustering_models.pkl",
                "embedding_model/model_info.json",
                "faiss_indices/",
                "training_report.json"
            ]
        }
        
        training_report_file = os.path.join(output_dir, 'training_report.json')
        with open(training_report_file, 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print(f"💾 Saved training report to {training_report_file}")
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETION SUCCESSFUL!")
        print(f"📊 Processed {len(key_profiles)} keys")
        print(f"🔍 Generated embeddings for {len(embeddings_data)} values")
        print(f"📁 All models saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during training completion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete AutoPatternChecker training")
    parser.add_argument("--output", default="trained_models", 
                       help="Output directory for trained models")
    
    args = parser.parse_args()
    complete_training(args.output)
