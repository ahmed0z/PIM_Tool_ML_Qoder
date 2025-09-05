"""
Unit tests for data ingestion module.
"""

import pytest
import pandas as pd
import tempfile
import os
from autopatternchecker.ingest import DataIngester

class TestDataIngester:
    """Test cases for DataIngester class."""
    
    def setup_method(self):
        """Set up test data."""
        self.config = {
            'key_columns': ['key_part1', 'key_part2', 'key_part3'],
            'value_column': 'value',
            'chunk_size': 1000,
            'encoding': 'utf-8'
        }
        self.ingester = DataIngester(self.config)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'key_part1': ['A', 'A', 'B', 'B', 'C'],
            'key_part2': ['X', 'X', 'Y', 'Y', 'Z'],
            'key_part3': ['1', '1', '2', '2', '3'],
            'value': ['value1', 'value2', 'value3', 'value4', 'value5']
        })
    
    def test_create_composite_keys(self):
        """Test composite key creation."""
        df = self.test_data.copy()
        result = self.ingester.create_composite_keys(df)
        
        assert '__composite_key__' in result.columns
        expected_keys = ['A||X||1', 'A||X||1', 'B||Y||2', 'B||Y||2', 'C||Z||3']
        assert result['__composite_key__'].tolist() == expected_keys
    
    def test_compute_key_statistics(self):
        """Test key statistics computation."""
        df = self.test_data.copy()
        df = self.ingester.create_composite_keys(df)
        
        stats_df = self.ingester.compute_key_statistics(df)
        
        assert len(stats_df) == 3  # Three unique composite keys
        assert 'composite_key' in stats_df.columns
        assert 'count' in stats_df.columns
        assert 'unique_values' in stats_df.columns
        
        # Check specific statistics
        ax1_stats = stats_df[stats_df['composite_key'] == 'A||X||1'].iloc[0]
        assert ax1_stats['count'] == 2
        assert ax1_stats['unique_values'] == 2
    
    def test_validate_data_schema(self):
        """Test data schema validation."""
        # Valid data
        result = self.ingester.validate_data_schema(self.test_data)
        assert result['is_valid'] == True
        assert len(result['issues']) == 0
        
        # Missing required columns
        invalid_df = self.test_data.drop('key_part1', axis=1)
        result = self.ingester.validate_data_schema(invalid_df)
        assert result['is_valid'] == False
        assert 'Missing required columns' in str(result['issues'])
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        result = self.ingester.validate_data_schema(empty_df)
        assert result['is_valid'] == False
        assert 'DataFrame is empty' in result['issues']
    
    def test_process_file(self):
        """Test file processing."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Process the file
            processed_df, key_stats_df = self.ingester.process_file(temp_path)
            
            # Check results
            assert len(processed_df) == len(self.test_data)
            assert '__composite_key__' in processed_df.columns
            assert len(key_stats_df) == 3  # Three unique keys
            
            # Check that all original data is preserved
            for col in self.test_data.columns:
                assert col in processed_df.columns
                
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_aggregate_key_statistics(self):
        """Test key statistics aggregation across chunks."""
        # Create mock chunk statistics
        chunk1_stats = pd.DataFrame([{
            'composite_key': 'A||X||1',
            'count': 5,
            'unique_values': 4,
            'unique_ratio': 0.8,
            'is_free_text': False,
            'top_signatures': [{'sig': 'L3', 'count': 3, 'pct': 60.0, 'examples': ['val1', 'val2']}],
            'sample_values': ['val1', 'val2']
        }])
        
        chunk2_stats = pd.DataFrame([{
            'composite_key': 'A||X||1',
            'count': 3,
            'unique_values': 2,
            'unique_ratio': 0.67,
            'is_free_text': False,
            'top_signatures': [{'sig': 'L3', 'count': 2, 'pct': 66.7, 'examples': ['val3', 'val4']}],
            'sample_values': ['val3', 'val4']
        }])
        
        combined_stats = pd.concat([chunk1_stats, chunk2_stats], ignore_index=True)
        aggregated = self.ingester._aggregate_key_statistics(combined_stats)
        
        assert len(aggregated) == 1
        ax1_stats = aggregated.iloc[0]
        assert ax1_stats['count'] == 8  # 5 + 3
        assert ax1_stats['unique_values'] == 6  # 4 + 2
        assert ax1_stats['composite_key'] == 'A||X||1'