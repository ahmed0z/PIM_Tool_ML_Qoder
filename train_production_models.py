#!/usr/bin/env python3
"""
Production model training script for AutoPatternChecker.
Creates all the expected output files and directories.

Usage: python3 train_production_models.py your_data.csv --output trained_models
"""

import argparse
import pandas as pd
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import logging

from autopatternchecker import (
    DataIngester, PatternProfiler, NormalizationEngine,
    ClusterAnalyzer, EmbeddingGenerator, FAISSIndexer, ModelTrainer
)
from autopatternchecker.utils import setup_logging, save_json

def create_output_structure(output_dir):
    """Create the expected output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_path / "embedding_model").mkdir(exist_ok=True)
    (output_path / "faiss_indices").mkdir(exist_ok=True)
    
    return output_path

def train_production_models(csv_file, output_dir, config=None):
    """Train production models and save all outputs."""
    
    print(f"🚀 Training AutoPatternChecker Production Models")
    print(f"📁 Data file: {csv_file}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output structure
    output_path = create_output_structure(output_dir)
    
    # Default configuration optimized for small datasets
    if config is None:
        config = {
            'key_columns': ['key_part1', 'key_part2', 'key_part3'],
            'value_column': 'value',
            'chunk_size': 10000,
            'encoding': 'utf-8',
            'min_signature_frequency': 2,  # Lower threshold for small datasets
            'rare_signature_pct_threshold': 5.0,  # Higher threshold for small datasets
            'tfidf_ngram_range': [1, 3],  # Smaller n-grams for small datasets
            'tfidf_max_features': 500,  # Fewer features for small datasets
            'hdbscan_min_cluster_size': 2,  # Lower cluster size for small datasets
            'hdbscan_min_samples': 1,
            'kmeans_n_clusters': 4,  # Fewer clusters for small datasets
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_batch_size': 32,
            'use_embeddings': True,
            'index_type': 'flat',
            'metric': 'euclidean'
        }
    
    # Step 1: Data Ingestion
    print("\n🔄 Step 1: Data Ingestion and Processing...")
    ingester = DataIngester(config)
    processed_df, key_stats_df = ingester.process_file(csv_file)
    
    print(f"✅ Processed {len(processed_df)} rows")
    print(f"✅ Found {len(key_stats_df)} unique composite keys")
    
    # Step 2: Pattern Profiling
    print("\n🔄 Step 2: Pattern Analysis and Profiling...")
    profiler = PatternProfiler(config)
    key_profiles = profiler.analyze_key_patterns(key_stats_df, processed_df)
    
    print(f"✅ Generated profiles for {len(key_profiles)} keys")
    
    # Save key profiles
    profiles_file = output_path / "key_profiles.json"
    save_json(key_profiles, profiles_file)
    print(f"💾 Saved key profiles to {profiles_file}")
    
    # Step 3: Normalization Rules
    print("\n🔄 Step 3: Normalization Rule Generation...")
    normalizer = NormalizationEngine(config)
    
    # Generate normalization rules from the profiles
    normalization_rules = normalizer.generate_normalization_rules(key_profiles)
    
    print(f"✅ Generated normalization rules for {len(normalization_rules)} keys")
    
    # Save normalization rules
    rules_file = output_path / "normalization_rules.json"
    save_json(normalization_rules, rules_file)
    print(f"💾 Saved normalization rules to {rules_file}")
    
    # Step 4: Clustering Models
    print("\n🔄 Step 4: Training Clustering Models...")
    cluster_analyzer = ClusterAnalyzer(config)
    
    clustering_models = {}
    for key, profile in key_profiles.items():
        if profile.get('sample_values') and len(profile['sample_values']) >= 2:
            try:
                model_data = cluster_analyzer.cluster_key_values(key, profile['sample_values'], profile)
                if model_data:
                    clustering_models[key] = model_data
            except Exception as e:
                logging.warning(f"Could not train clustering for key {key}: {e}")
    
    print(f"✅ Trained clustering models for {len(clustering_models)} keys")
    
    # Save clustering models
    clustering_file = output_path / "clustering_models.pkl"
    with open(clustering_file, 'wb') as f:
        pickle.dump(clustering_models, f)
    print(f"💾 Saved clustering models to {clustering_file}")
    
    # Step 5: Embedding Models and FAISS Indices
    print("\n🔄 Step 5: Training Embedding Models and Building Indices...")
    embedding_gen = EmbeddingGenerator(config)
    faiss_indexer = FAISSIndexer(config)
    
    # Collect all values for embedding training
    all_values = []
    value_to_key = {}
    
    for key, profile in key_profiles.items():
        if profile.get('sample_values'):
            for value in profile['sample_values']:
                all_values.append(value)
                value_to_key[value] = key
    
    if all_values:
        print(f"📊 Processing {len(all_values)} values for embeddings...")
        
        # Generate embeddings
        embeddings = embedding_gen.generate_embeddings(all_values)
        
        # Build FAISS index
        index = faiss_indexer.create_index(embeddings, embeddings.shape[1])
        
        # Save embedding model info
        embedding_info = {
            'model_name': config['embedding_model'],
            'dimension': embeddings.shape[1],
            'total_vectors': len(embeddings),
            'values': all_values,
            'value_to_key_mapping': value_to_key
        }
        
        embedding_model_file = output_path / "embedding_model" / "model_info.json"
        save_json(embedding_info, embedding_model_file)
        
        # Save FAISS indices
        faiss_indices_dir = output_path / "faiss_indices"
        faiss_indexer.save_indices(str(faiss_indices_dir))
        
        print(f"✅ Built embeddings for {len(all_values)} values")
        print(f"💾 Saved embedding info to {embedding_model_file}")
        print(f"💾 Saved FAISS indices to {faiss_indices_dir}")
    else:
        print("⚠️  No values available for embedding training")
    
    # Step 6: Training Report
    print("\n🔄 Step 6: Generating Training Report...")
    
    training_report = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(csv_file),
        "output_directory": str(output_dir),
        "config": config,
        "data_stats": {
            "total_rows": len(processed_df),
            "unique_keys": len(key_stats_df),
            "processed_values": len(all_values) if 'all_values' in locals() else 0
        },
        "model_stats": {
            "key_profiles": len(key_profiles),
            "normalization_rules": len(normalization_rules),
            "clustering_models": len(clustering_models),
            "embedding_vectors": len(all_values) if 'all_values' in locals() else 0
        },
        "files_created": [
            "key_profiles.json",
            "normalization_rules.json", 
            "clustering_models.pkl",
            "embedding_model/model_info.json",
            "faiss_indices/main_index.faiss",
            "training_report.json"
        ]
    }
    
    report_file = output_path / "training_report.json"
    save_json(training_report, report_file)
    print(f"💾 Saved training report to {report_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Processed {training_report['data_stats']['total_rows']} rows")
    print(f"🔑 Generated {training_report['model_stats']['key_profiles']} key profiles")
    print(f"📋 Created {training_report['model_stats']['normalization_rules']} normalization rule sets")
    print(f"🤖 Trained {training_report['model_stats']['clustering_models']} clustering models")
    print(f"🔍 Built embeddings for {training_report['model_stats']['embedding_vectors']} values")
    print(f"📁 All models saved to: {output_path}")
    print("=" * 60)
    
    return training_report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train AutoPatternChecker production models')
    parser.add_argument('data_file', help='Path to CSV data file')
    parser.add_argument('--output', default='./trained_models', help='Output directory for models')
    parser.add_argument('--config', help='Path to config JSON file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Train models
    try:
        report = train_production_models(args.data_file, args.output, config)
        print(f"\n✅ Training completed successfully!")
        print(f"📋 See {args.output}/training_report.json for details")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.exception("Training failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
