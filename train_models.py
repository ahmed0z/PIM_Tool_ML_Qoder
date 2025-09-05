#!/usr/bin/env python3
"""
Simple script to train AutoPatternChecker models.
Usage: python train_models.py --data your_data.csv --output models/
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from autopatternchecker import (
    DataIngester, PatternProfiler, ModelTrainer
)
from autopatternchecker.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Train AutoPatternChecker models')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--output', default='./trained_models', help='Output directory for models')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--key-columns', nargs=3, default=['key_part1', 'key_part2', 'key_part3'],
                       help='Key column names')
    parser.add_argument('--value-column', default='value', help='Value column name')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    # Configuration
    config = {
        'key_columns': args.key_columns,
        'value_column': args.value_column,
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
        'metric': 'cosine'
    }
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    print("🚀 Starting AutoPatternChecker Model Training...")
    print(f"📁 Data file: {args.data}")
    print(f"📁 Output directory: {args.output}")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process data
    print("\n🔄 Step 1: Processing data...")
    ingester = DataIngester(config)
    processed_df, key_stats_df = ingester.process_file(args.data)
    
    print(f"✅ Processed {len(processed_df)} rows")
    print(f"✅ Found {len(key_stats_df)} unique composite keys")
    
    # Step 2: Generate profiles
    print("\n🔄 Step 2: Generating profiles...")
    profiler = PatternProfiler(config)
    key_profiles = profiler.analyze_key_patterns(key_stats_df, processed_df)
    
    print(f"✅ Generated profiles for {len(key_profiles)} keys")
    
    # Step 3: Train models
    print("\n🔄 Step 3: Training models...")
    trainer = ModelTrainer(config)
    
    # Prepare training data
    training_data = trainer.prepare_training_data(key_profiles, processed_df)
    print(f"✅ Prepared training data for {len(training_data['clustering_data'])} keys")
    
    # Train clustering models
    print("  Training clustering models...")
    clustering_models = trainer.train_clustering_models(training_data)
    print(f"  ✅ Trained clustering for {len(clustering_models)} keys")
    
    # Train embedding models
    print("  Training embedding models...")
    embedding_models = trainer.train_embedding_models(training_data)
    if embedding_models:
        print(f"  ✅ Trained embedding model with {embedding_models['training_examples']} examples")
    else:
        print("  ⚠️  No embedding training data available")
    
    # Train normalization models
    print("  Training normalization models...")
    normalization_models = trainer.train_normalization_models(training_data)
    if normalization_models:
        print(f"  ✅ Trained normalization with {len(normalization_models['rules'])} rules")
    else:
        print("  ⚠️  No normalization training data available")
    
    # Step 4: Save models
    print("\n🔄 Step 4: Saving models...")
    trainer.save_trained_models(args.output)
    
    # Step 5: Generate report
    print("\n🔄 Step 5: Generating training report...")
    summary = trainer.get_training_summary()
    
    report = {
        'training_date': pd.Timestamp.now().isoformat(),
        'data_file': args.data,
        'config': config,
        'summary': summary,
        'data_stats': {
            'total_rows': len(processed_df),
            'unique_keys': len(key_stats_df),
            'training_keys': len(training_data['clustering_data'])
        }
    }
    
    report_path = f'{args.output}/training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Training report saved to {report_path}")
    
    # Final summary
    print("\n🎉 Training completed successfully!")
    print(f"📊 Models trained: {summary['models_trained']}")
    print(f"📁 Models saved to: {args.output}")
    print(f"📋 Report saved to: {report_path}")

if __name__ == "__main__":
    main()