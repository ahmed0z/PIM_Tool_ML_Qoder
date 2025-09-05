"""
Unit tests for AutoPatternChecker utilities.
"""

import pytest
import pandas as pd
import numpy as np
from autopatternchecker.utils import (
    create_composite_key, generate_signature, signature_to_regex,
    normalize_whitespace, remove_space_between_number_and_unit,
    normalize_thousands_separators, extract_parenthetical_comments,
    calculate_similarity_stats, validate_composite_key_format
)

class TestUtils:
    """Test cases for utility functions."""
    
    def test_create_composite_key(self):
        """Test composite key creation."""
        row = pd.Series({'key1': 'A', 'key2': 'B', 'key3': 'C'})
        key_columns = ['key1', 'key2', 'key3']
        result = create_composite_key(row, key_columns)
        assert result == 'A||B||C'
    
    def test_generate_signature(self):
        """Test signature generation."""
        # Test basic signature
        assert generate_signature('ABC123') == 'L3D3'
        assert generate_signature('Hello World') == 'L5_L5'
        assert generate_signature('123.45') == 'D3OD2'
        assert generate_signature('') == ''
        assert generate_signature(None) == ''
    
    def test_signature_to_regex(self):
        """Test signature to regex conversion."""
        # Test basic patterns
        assert signature_to_regex('L3D3') == '[A-Za-z\\u0600-\\u06FF]{3}\\d{3}'
        assert signature_to_regex('D+') == '\\d+'
        assert signature_to_regex('L+') == '[A-Za-z\\u0600-\\u06FF]+'
        assert signature_to_regex('') == '.*'
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_whitespace('  hello   world  ') == 'hello world'
        assert normalize_whitespace('hello\nworld') == 'hello world'
        assert normalize_whitespace('') == ''
        assert normalize_whitespace(None) == ''
    
    def test_remove_space_between_number_and_unit(self):
        """Test space removal between numbers and units."""
        assert remove_space_between_number_and_unit('1 V') == '1V'
        assert remove_space_between_number_and_unit('100 Hz') == '100Hz'
        assert remove_space_between_number_and_unit('hello world') == 'hello world'
        assert remove_space_between_number_and_unit('') == ''
    
    def test_normalize_thousands_separators(self):
        """Test thousands separator normalization."""
        assert normalize_thousands_separators('1,000') == '1000'
        assert normalize_thousands_separators('1,000,000') == '1000000'
        assert normalize_thousands_separators('hello,world') == 'hello,world'
        assert normalize_thousands_separators('') == ''
    
    def test_extract_parenthetical_comments(self):
        """Test parenthetical comment extraction."""
        value, comment = extract_parenthetical_comments('Hello (world)')
        assert value == 'Hello'
        assert comment == 'world'
        
        value, comment = extract_parenthetical_comments('Hello world')
        assert value == 'Hello world'
        assert comment is None
        
        value, comment = extract_parenthetical_comments('')
        assert value == ''
        assert comment is None
    
    def test_calculate_similarity_stats(self):
        """Test similarity statistics calculation."""
        similarities = [0.8, 0.9, 0.7, 0.85, 0.75]
        stats = calculate_similarity_stats(similarities)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert stats['min'] == 0.75
        assert stats['max'] == 0.9
        
        # Test empty list
        empty_stats = calculate_similarity_stats([])
        assert empty_stats['mean'] == 0.0
    
    def test_validate_composite_key_format(self):
        """Test composite key format validation."""
        # Valid format
        assert validate_composite_key_format(['A', 'B', 'C'], 3) == True
        assert validate_composite_key_format(['key1', 'key2', 'key3'], 3) == True
        
        # Invalid format
        assert validate_composite_key_format(['A', 'B'], 3) == False  # Wrong length
        assert validate_composite_key_format(['A', '', 'C'], 3) == False  # Empty part
        assert validate_composite_key_format([None, 'B', 'C'], 3) == False  # None part