"""
Data ingestion module for AutoPatternChecker.
Handles CSV reading, composite key creation, and initial data profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterator
from pathlib import Path
import logging
from .utils import create_composite_key, generate_signature, chunk_dataframe

logger = logging.getLogger(__name__)

class DataIngester:
    """Handles data ingestion and initial processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_columns = config.get('key_columns', ['key_part1', 'key_part2', 'key_part3'])
        self.value_column = config.get('value_column', 'value')
        self.chunk_size = config.get('chunk_size', 10000)
        self.encoding = config.get('encoding', 'utf-8')
        
    def read_csv_streaming(self, filepath: str) -> Iterator[pd.DataFrame]:
        """Read CSV file in streaming mode with chunking."""
        try:
            for chunk in pd.read_csv(
                filepath, 
                chunksize=self.chunk_size,
                encoding=self.encoding,
                dtype=str  # Read all as strings initially
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
    
    def create_composite_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite keys for the dataframe."""
        if '__composite_key__' in df.columns:
            logger.info("Composite key column already exists")
            return df
        
        # Validate that all key columns exist
        missing_cols = [col for col in self.key_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing key columns: {missing_cols}")
        
        # Create composite key
        df['__composite_key__'] = df.apply(
            lambda row: create_composite_key(row, self.key_columns), 
            axis=1
        )
        
        logger.info(f"Created composite keys for {len(df)} rows")
        return df
    
    def compute_key_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic statistics for each composite key."""
        stats = []
        
        for composite_key, group in df.groupby('__composite_key__'):
            values = group[self.value_column].dropna()
            
            if len(values) == 0:
                continue
            
            # Generate signatures for all values
            signatures = values.apply(generate_signature)
            signature_counts = signatures.value_counts()
            
            # Get top signatures
            top_signatures = []
            for sig, count in signature_counts.head(10).items():
                pct = (count / len(values)) * 100
                examples = values[signatures == sig].head(3).tolist()
                top_signatures.append({
                    "sig": sig,
                    "count": int(count),
                    "pct": round(pct, 2),
                    "examples": examples
                })
            
            # Check if this looks like free-text (high uniqueness)
            unique_ratio = len(values.unique()) / len(values)
            is_free_text = unique_ratio > 0.8
            
            stats.append({
                'composite_key': composite_key,
                'count': len(values),
                'unique_values': len(values.unique()),
                'unique_ratio': round(unique_ratio, 3),
                'is_free_text': is_free_text,
                'top_signatures': top_signatures,
                'sample_values': values.head(5).tolist()
            })
        
        return pd.DataFrame(stats)
    
    def process_file(self, filepath_or_fileobj) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process a CSV file and return both the processed data and key statistics.

        Args:
            filepath_or_fileobj: Either a file path string or a file-like object (e.g., Streamlit UploadedFile)

        Returns:
            Tuple of (processed_df, key_stats_df)
        """
        logger.info(f"Processing file: {filepath_or_fileobj}")

        # Handle different input types
        if hasattr(filepath_or_fileobj, 'read'):
            # File-like object (e.g., Streamlit UploadedFile)
            return self._process_file_object(filepath_or_fileobj)
        else:
            # File path string
            return self._process_file_path(filepath_or_fileobj)

    def _process_file_path(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a file from a file path."""
        # Read and process in chunks
        all_chunks = []
        key_stats_list = []

        for chunk in self.read_csv_streaming(filepath):
            # Create composite keys
            chunk = self.create_composite_keys(chunk)

            # Compute statistics for this chunk
            chunk_stats = self.compute_key_statistics(chunk)
            key_stats_list.append(chunk_stats)

            all_chunks.append(chunk)

        # Combine all chunks
        processed_df = pd.concat(all_chunks, ignore_index=True)

        # Combine and aggregate key statistics
        if key_stats_list:
            combined_stats = pd.concat(key_stats_list, ignore_index=True)
            # Aggregate statistics across chunks
            key_stats_df = self._aggregate_key_statistics(combined_stats)
        else:
            key_stats_df = pd.DataFrame()

        logger.info(f"Processed {len(processed_df)} rows with {len(key_stats_df)} unique keys")
        return processed_df, key_stats_df

    def _process_file_object(self, file_obj) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a file from a file-like object (e.g., Streamlit UploadedFile)."""
        try:
            # Reset file pointer to beginning
            file_obj.seek(0)

            # Read the entire file into a DataFrame
            # For file objects, we read all at once rather than chunking
            df = pd.read_csv(file_obj, dtype=str, encoding=self.encoding)

            # Validate schema
            validation = self.validate_data_schema(df)
            if not validation['is_valid']:
                raise ValueError(f"Data validation failed: {validation['issues']}")

            # Create composite keys
            df = self.create_composite_keys(df)

            # Compute statistics
            key_stats_df = self.compute_key_statistics(df)

            logger.info(f"Processed {len(df)} rows with {len(key_stats_df)} unique keys")
            return df, key_stats_df

        except Exception as e:
            logger.error(f"Error processing file object: {e}")
            raise
    
    def _aggregate_key_statistics(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate key statistics across multiple chunks."""
        if stats_df.empty:
            return stats_df
        
        aggregated = []
        
        for composite_key, group in stats_df.groupby('composite_key'):
            # Sum counts
            total_count = group['count'].sum()
            total_unique = group['unique_values'].sum()
            
            # Combine top signatures
            all_signatures = []
            for _, row in group.iterrows():
                all_signatures.extend(row['top_signatures'])
            
            # Aggregate signature counts
            sig_counts = {}
            for sig_data in all_signatures:
                sig = sig_data['sig']
                if sig in sig_counts:
                    sig_counts[sig]['count'] += sig_data['count']
                    sig_counts[sig]['examples'].extend(sig_data['examples'])
                else:
                    sig_counts[sig] = {
                        'sig': sig,
                        'count': sig_data['count'],
                        'examples': sig_data['examples'][:]
                    }
            
            # Calculate percentages and get top signatures
            top_signatures = []
            for sig_data in sorted(sig_counts.values(), key=lambda x: x['count'], reverse=True)[:10]:
                pct = (sig_data['count'] / total_count) * 100
                top_signatures.append({
                    "sig": sig_data['sig'],
                    "count": sig_data['count'],
                    "pct": round(pct, 2),
                    "examples": list(set(sig_data['examples']))[:3]  # Remove duplicates
                })
            
            # Determine if free text
            unique_ratio = total_unique / total_count if total_count > 0 else 0
            is_free_text = unique_ratio > 0.8
            
            # Combine sample values
            all_samples = []
            for _, row in group.iterrows():
                all_samples.extend(row['sample_values'])
            sample_values = list(set(all_samples))[:5]  # Remove duplicates
            
            aggregated.append({
                'composite_key': composite_key,
                'count': total_count,
                'unique_values': total_unique,
                'unique_ratio': round(unique_ratio, 3),
                'is_free_text': is_free_text,
                'top_signatures': top_signatures,
                'sample_values': sample_values
            })
        
        return pd.DataFrame(aggregated)
    
    def validate_data_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that the data matches expected schema."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = self.key_columns + [self.value_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check for empty dataframe
        if len(df) == 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
        
        # Check for completely null values in key columns
        for col in self.key_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    validation_results['warnings'].append(
                        f"Column '{col}' has {null_count} null values"
                    )
        
        # Check for completely null values in value column
        if self.value_column in df.columns:
            null_count = df[self.value_column].isnull().sum()
            if null_count > 0:
                validation_results['warnings'].append(
                    f"Value column '{self.value_column}' has {null_count} null values"
                )
        
        return validation_results