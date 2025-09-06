#!/usr/bin/env python3
"""
Handle large CSV files for AutoPatternChecker
This script processes large files in chunks to avoid memory issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from autopatternchecker import DataIngester, PatternProfiler, ModelTrainer
from autopatternchecker.utils import setup_logging

def process_large_csv(input_file, output_dir="./large_file_output", chunk_size=10000):
    """
    Process a large CSV file in chunks to avoid memory issues.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed results
        chunk_size: Number of rows per chunk
    """
    
    print(f"🚀 Processing large file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'key_columns': ['key_part1', 'key_part2', 'key_part3'],
        'value_column': 'value',
        'chunk_size': chunk_size,
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
    
    # Get file size
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Process in chunks
    print("🔄 Processing file in chunks...")
    
    all_processed_chunks = []
    all_key_stats = []
    
    chunk_count = 0
    total_rows = 0
    
    # Read and process chunks
    for chunk_df in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk_df)
        
        print(f"  Processing chunk {chunk_count} ({len(chunk_df)} rows)...")
        
        # Process this chunk
        ingester = DataIngester(config)
        
        # Save chunk to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        chunk_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            # Process the chunk
            processed_chunk, key_stats_chunk = ingester.process_file(temp_file.name)
            
            # Store results
            all_processed_chunks.append(processed_chunk)
            all_key_stats.append(key_stats_chunk)
            
            print(f"    ✅ Chunk {chunk_count} processed successfully")
            
        except Exception as e:
            print(f"    ❌ Error processing chunk {chunk_count}: {e}")
            continue
        
        finally:
            # Clean up temp file
            os.unlink(temp_file.name)
    
    print(f"✅ Processed {chunk_count} chunks, {total_rows} total rows")
    
    # Combine all processed chunks
    print("🔄 Combining processed chunks...")
    processed_df = pd.concat(all_processed_chunks, ignore_index=True)
    
    # Combine key statistics
    print("🔄 Combining key statistics...")
    key_stats_df = pd.concat(all_key_stats, ignore_index=True)
    
    # Aggregate key statistics (sum counts, keep other stats)
    key_stats_aggregated = key_stats_df.groupby('composite_key').agg({
        'count': 'sum',
        'unique_values': 'sum',
        'unique_ratio': 'mean',
        'is_free_text': 'any',
        'is_numeric_unit': 'any'
    }).reset_index()
    
    # Save results
    print("💾 Saving results...")
    
    # Save processed data
    processed_file = f"{output_dir}/processed_data.csv"
    processed_df.to_csv(processed_file, index=False)
    print(f"  ✅ Processed data saved: {processed_file}")
    
    # Save key statistics
    key_stats_file = f"{output_dir}/key_statistics.csv"
    key_stats_aggregated.to_csv(key_stats_file, index=False)
    print(f"  ✅ Key statistics saved: {key_stats_file}")
    
    # Generate profiles
    print("🔍 Generating pattern profiles...")
    profiler = PatternProfiler(config)
    key_profiles = profiler.analyze_key_patterns(key_stats_aggregated, processed_df)
    
    # Save profiles
    import json
    profiles_file = f"{output_dir}/key_profiles.json"
    with open(profiles_file, 'w') as f:
        json.dump(key_profiles, f, indent=2, default=str)
    print(f"  ✅ Key profiles saved: {profiles_file}")
    
    # Train models
    print("🤖 Training models...")
    trainer = ModelTrainer(config)
    training_data = trainer.prepare_training_data(key_profiles, processed_df)
    
    # Train clustering models
    clustering_models = trainer.train_clustering_models(training_data)
    print(f"  ✅ Clustering models trained for {len(clustering_models)} keys")
    
    # Train embedding models
    embedding_models = trainer.train_embedding_models(training_data)
    if embedding_models:
        print(f"  ✅ Embedding model trained with {embedding_models['training_examples']} examples")
    
    # Train normalization models
    normalization_models = trainer.train_normalization_models(training_data)
    if normalization_models:
        print(f"  ✅ Normalization model trained with {len(normalization_models['rules'])} rules")
    
    # Save trained models
    models_dir = f"{output_dir}/trained_models"
    trainer.save_trained_models(models_dir)
    print(f"  ✅ Trained models saved: {models_dir}")
    
    # Generate summary report
    summary = {
        'input_file': input_file,
        'file_size_mb': file_size_mb,
        'total_rows': total_rows,
        'chunks_processed': chunk_count,
        'unique_keys': len(key_stats_aggregated),
        'models_trained': list(trainer.trained_models.keys()),
        'output_directory': output_dir
    }
    
    summary_file = f"{output_dir}/processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✅ Summary report saved: {summary_file}")
    
    print(f"\n🎉 Large file processing completed!")
    print(f"📊 Results:")
    print(f"  - Total rows processed: {total_rows:,}")
    print(f"  - Unique keys found: {len(key_stats_aggregated):,}")
    print(f"  - Models trained: {len(trainer.trained_models)}")
    print(f"  - Output directory: {output_dir}")
    
    return processed_df, key_stats_aggregated, key_profiles, trainer

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process large CSV files for AutoPatternChecker')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', default='./large_file_output', help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Process the file
    try:
        processed_df, key_stats, profiles, trainer = process_large_csv(
            args.input_file, 
            args.output, 
            args.chunk_size
        )
        print("\n✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())