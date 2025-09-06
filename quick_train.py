#!/usr/bin/env python3
"""
Quick training script for AutoPatternChecker models.
Run this in Colab or any Python environment.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import AutoPatternChecker
from autopatternchecker import (
    DataIngester, PatternProfiler, ModelTrainer
)
from autopatternchecker.utils import setup_logging

def quick_train(csv_path, output_dir="./trained_models"):
    """Quick training function for AutoPatternChecker models."""
    
    print("🚀 Starting AutoPatternChecker Quick Training...")
    
    # Setup logging
    setup_logging('INFO')
    
    # Configuration
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
        'metric': 'cosine'
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process data
    print("\n🔄 Step 1: Processing data...")
    ingester = DataIngester(config)
    processed_df, key_stats_df = ingester.process_file(csv_path)
    
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
    trainer.save_trained_models(output_dir)
    
    # Step 5: Generate report
    print("\n🔄 Step 5: Generating training report...")
    summary = trainer.get_training_summary()
    
    report = {
        'training_date': datetime.now().isoformat(),
        'data_file': csv_path,
        'config': config,
        'summary': summary,
        'data_stats': {
            'total_rows': len(processed_df),
            'unique_keys': len(key_stats_df),
            'training_keys': len(training_data['clustering_data'])
        }
    }
    
    report_path = f'{output_dir}/training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Training report saved to {report_path}")
    
    # Final summary
    print("\n🎉 Training completed successfully!")
    print(f"📊 Models trained: {summary['models_trained']}")
    print(f"📁 Models saved to: {output_dir}")
    print(f"📋 Report saved to: {report_path}")
    
    return trainer, report

# Example usage
if __name__ == "__main__":
    # Example: Train on sample data
    print("Creating sample data...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'key_part1': ['Tools', 'Tools', 'Electronics', 'Electronics', 'Tools'] * 20,
        'key_part2': ['Accessories', 'Accessories', 'Components', 'Components', 'Accessories'] * 20,
        'key_part3': ['Type A', 'Type A', 'Resistor', 'Capacitor', 'Type B'] * 20,
        'value': [
            'SMB Mini Jack Right Angle', 'SC Plug', '1kΩ 1/4W', '100µF 25V', 'SMB Mini Jack',
            'SC Plug Right Angle', '2.2kΩ 1/2W', '220µF 50V', 'SMB Mini Jack Straight',
            'SC Plug Straight', '10kΩ 1/4W', '47µF 16V', 'SMB Mini Jack Left Angle',
            'SC Plug Left Angle', '100Ω 1/4W', '1000µF 35V', 'SMB Mini Jack Right',
            'SC Plug Right', '4.7kΩ 1/2W', '470µF 25V'
        ] * 5
    })
    
    # Save sample data
    sample_path = 'sample_data.csv'
    sample_data.to_csv(sample_path, index=False)
    print(f"✅ Sample data created: {sample_path}")
    
    # Train models
    trainer, report = quick_train(sample_path, './quick_trained_models')
    
    print("\n🎯 Training complete! You can now use the trained models in your AutoPatternChecker system.")