"""
Pattern profiling module for AutoPatternChecker.
Handles signature analysis, regex generation, and key profiling.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from .utils import generate_signature, signature_to_regex

logger = logging.getLogger(__name__)

class PatternProfiler:
    """Handles pattern analysis and profiling for composite keys."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_signature_frequency = config.get('min_signature_frequency', 5)
        self.rare_signature_pct_threshold = config.get('rare_signature_pct_threshold', 1.0)
        
    def analyze_key_patterns(self, key_stats_df: pd.DataFrame, values_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns for all keys and generate profiles.
        
        Args:
            key_stats_df: DataFrame with key statistics
            values_df: DataFrame with all values (must have __composite_key__ and value columns)
        
        Returns:
            Dictionary with key profiles
        """
        key_profiles = {}
        
        for _, key_row in key_stats_df.iterrows():
            composite_key = key_row['composite_key']
            
            # Get all values for this key
            key_values = values_df[values_df['__composite_key__'] == composite_key]['value'].dropna()
            
            if len(key_values) == 0:
                continue
            
            # Generate profile for this key
            profile = self._generate_key_profile(composite_key, key_values, key_row)
            key_profiles[composite_key] = profile
            
        logger.info(f"Generated profiles for {len(key_profiles)} keys")
        return key_profiles
    
    def _generate_key_profile(self, composite_key: str, values: pd.Series, key_stats: pd.Series) -> Dict[str, Any]:
        """Generate a detailed profile for a single key."""
        
        # Generate signatures for all values
        signatures = values.apply(generate_signature)
        signature_counts = signatures.value_counts()
        
        # Filter out rare signatures
        total_values = len(values)
        min_count = max(self.min_signature_frequency, int(total_values * self.rare_signature_pct_threshold / 100))
        common_signatures = signature_counts[signature_counts >= min_count]
        
        # Generate top signatures with examples
        top_signatures = []
        for sig, count in common_signatures.head(10).items():
            pct = (count / total_values) * 100
            examples = values[signatures == sig].head(3).tolist()
            top_signatures.append({
                "sig": sig,
                "count": int(count),
                "pct": round(pct, 2),
                "examples": examples
            })
        
        # Generate candidate regex from most common signature
        candidate_regex = None
        if top_signatures:
            most_common_sig = top_signatures[0]['sig']
            candidate_regex = signature_to_regex(most_common_sig)
        
        # Analyze value characteristics
        value_analysis = self._analyze_value_characteristics(values)
        
        # Determine if this is a numeric/unit pattern
        is_numeric_unit = self._is_numeric_unit_pattern(values)
        
        profile = {
            "composite_key": composite_key,
            "count": int(key_stats['count']),
            "unique_values": int(key_stats['unique_values']),
            "unique_ratio": key_stats['unique_ratio'],
            "is_free_text": key_stats['is_free_text'],
            "is_numeric_unit": is_numeric_unit,
            "top_signatures": top_signatures,
            "candidate_regex": candidate_regex,
            "value_analysis": value_analysis,
            "sample_values": key_stats['sample_values']
        }
        
        return profile
    
    def _analyze_value_characteristics(self, values: pd.Series) -> Dict[str, Any]:
        """Analyze characteristics of values for a key."""
        if len(values) == 0:
            return {}
        
        # Basic statistics
        lengths = values.str.len()
        
        # Character type analysis
        has_digits = values.str.contains(r'\d', na=False).sum()
        has_letters = values.str.contains(r'[A-Za-z]', na=False).sum()
        has_special = values.str.contains(r'[^A-Za-z0-9\s]', na=False).sum()
        has_whitespace = values.str.contains(r'\s', na=False).sum()
        
        # Pattern analysis
        has_parentheses = values.str.contains(r'\([^)]+\)', na=False).sum()
        has_commas = values.str.contains(r',', na=False).sum()
        has_dashes = values.str.contains(r'-', na=False).sum()
        
        # Unit patterns
        unit_patterns = [
            r'\b(V|A|W|Hz|MHz|GHz|kHz|Ω|Ohm|F|H|C|K|J)\b',
            r'\b(volts?|amps?|watts?|hertz|ohms?|farads?|henries?|celsius|kelvin|joules?)\b'
        ]
        has_units = any(values.str.contains(pattern, case=False, na=False).sum() > 0 for pattern in unit_patterns)
        
        total = len(values)
        
        return {
            "length_stats": {
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "median": float(lengths.median())
            },
            "character_types": {
                "has_digits": int(has_digits),
                "has_letters": int(has_letters),
                "has_special": int(has_special),
                "has_whitespace": int(has_whitespace),
                "pct_digits": round((has_digits / total) * 100, 2),
                "pct_letters": round((has_letters / total) * 100, 2),
                "pct_special": round((has_special / total) * 100, 2),
                "pct_whitespace": round((has_whitespace / total) * 100, 2)
            },
            "patterns": {
                "has_parentheses": int(has_parentheses),
                "has_commas": int(has_commas),
                "has_dashes": int(has_dashes),
                "has_units": bool(has_units),
                "pct_parentheses": round((has_parentheses / total) * 100, 2),
                "pct_commas": round((has_commas / total) * 100, 2),
                "pct_dashes": round((has_dashes / total) * 100, 2)
            }
        }
    
    def _is_numeric_unit_pattern(self, values: pd.Series) -> bool:
        """Determine if values follow a numeric + unit pattern."""
        if len(values) == 0:
            return False
        
        # Pattern: number (with optional decimal) + optional space + unit
        numeric_unit_pattern = r'^\s*\d+(?:\.\d+)?\s*[A-Za-z]+\s*$'
        
        matches = values.str.match(numeric_unit_pattern, na=False)
        match_ratio = matches.sum() / len(values)
        
        # Consider it a numeric-unit pattern if >70% match
        return match_ratio > 0.7
    
    def refine_regex_pattern(self, values: pd.Series, base_regex: str) -> str:
        """
        Refine a regex pattern using actual value examples.
        This is a basic implementation that can be enhanced.
        """
        if not base_regex or len(values) == 0:
            return base_regex
        
        # For now, return the base regex
        # In a more sophisticated implementation, this would:
        # 1. Test the regex against all values
        # 2. Identify common patterns in non-matching values
        # 3. Adjust the regex accordingly
        # 4. Handle edge cases and special characters
        
        return base_regex
    
    def generate_normalization_hints(self, profile: Dict[str, Any]) -> List[str]:
        """Generate hints for normalization rules based on profile analysis."""
        hints = []
        
        value_analysis = profile.get('value_analysis', {})
        patterns = value_analysis.get('patterns', {})
        char_types = value_analysis.get('character_types', {})
        
        # Whitespace normalization
        if patterns.get('pct_whitespace', 0) > 10:
            hints.append('trim_whitespace')
        
        # Parenthetical comments
        if patterns.get('pct_parentheses', 0) > 5:
            hints.append('extract_parenthetical_comments')
        
        # Comma handling (thousands separators)
        if patterns.get('pct_commas', 0) > 10:
            hints.append('normalize_thousands_separators')
        
        # Unit spacing
        if profile.get('is_numeric_unit', False):
            hints.append('remove_space_between_number_and_unit')
        
        # Case normalization (if mostly letters)
        if char_types.get('pct_letters', 0) > 80:
            hints.append('case_normalization')
        
        return hints
    
    def validate_pattern_coverage(self, profile: Dict[str, Any], test_values: List[str]) -> Dict[str, Any]:
        """Validate how well the pattern covers test values."""
        candidate_regex = profile.get('candidate_regex')
        if not candidate_regex:
            return {"coverage": 0.0, "matches": [], "non_matches": test_values}
        
        try:
            pattern = re.compile(candidate_regex)
            matches = []
            non_matches = []
            
            for value in test_values:
                if pattern.match(str(value)):
                    matches.append(value)
                else:
                    non_matches.append(value)
            
            coverage = len(matches) / len(test_values) if test_values else 0.0
            
            return {
                "coverage": round(coverage, 3),
                "matches": matches,
                "non_matches": non_matches,
                "total_tested": len(test_values)
            }
            
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {candidate_regex}, error: {e}")
            return {"coverage": 0.0, "matches": [], "non_matches": test_values, "error": str(e)}
    
    def suggest_pattern_improvements(self, profile: Dict[str, Any], validation_results: Dict[str, Any]) -> List[str]:
        """Suggest improvements to patterns based on validation results."""
        suggestions = []
        
        coverage = validation_results.get('coverage', 0.0)
        non_matches = validation_results.get('non_matches', [])
        
        if coverage < 0.8 and non_matches:
            # Analyze non-matching values
            non_match_signatures = [generate_signature(val) for val in non_matches]
            unique_signatures = set(non_match_signatures)
            
            if len(unique_signatures) <= 3:
                suggestions.append("Consider adding patterns for common non-matching signatures")
            else:
                suggestions.append("Many unique patterns in non-matches - may need clustering analysis")
        
        if coverage < 0.5:
            suggestions.append("Low pattern coverage - consider if this key needs different approach")
        
        return suggestions