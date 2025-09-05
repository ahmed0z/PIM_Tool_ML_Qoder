"""
Normalization module for AutoPatternChecker.
Handles value normalization using learned rules.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from .utils import (
    normalize_whitespace, 
    remove_space_between_number_and_unit,
    normalize_thousands_separators,
    extract_parenthetical_comments
)

logger = logging.getLogger(__name__)

class NormalizationEngine:
    """Handles value normalization using configurable rules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalization_rules = {}
        self.rule_functions = self._initialize_rule_functions()
        
    def _initialize_rule_functions(self) -> Dict[str, Callable]:
        """Initialize available normalization rule functions."""
        return {
            'trim_whitespace': self._trim_whitespace,
            'remove_space_between_number_and_unit': self._remove_space_between_number_and_unit,
            'normalize_thousands_separators': self._normalize_thousands_separators,
            'extract_parenthetical_comments': self._extract_parenthetical_comments,
            'case_normalization': self._case_normalization,
            'remove_extra_punctuation': self._remove_extra_punctuation,
            'normalize_units': self._normalize_units,
            'remove_leading_zeros': self._remove_leading_zeros
        }
    
    def load_normalization_rules(self, rules: Dict[str, Any]) -> None:
        """Load normalization rules for different keys."""
        self.normalization_rules = rules
        logger.info(f"Loaded normalization rules for {len(rules)} keys")
    
    def normalize_value(self, value: str, composite_key: str, 
                       return_metadata: bool = False) -> Any:
        """
        Normalize a value using rules for the given composite key.
        
        Args:
            value: The value to normalize
            composite_key: The composite key to get rules for
            return_metadata: Whether to return metadata about applied rules
        
        Returns:
            Normalized value, or tuple of (normalized_value, metadata) if return_metadata=True
        """
        if pd.isna(value) or value == "":
            return value if not return_metadata else (value, {})
        
        value = str(value)
        original_value = value
        applied_rules = []
        metadata = {}
        
        # Get rules for this key
        key_rules = self.normalization_rules.get(composite_key, {}).get('rules', [])
        
        # Apply rules in order
        for rule_config in key_rules:
            rule_name = rule_config.get('name')
            if rule_name in self.rule_functions:
                try:
                    value, rule_metadata = self.rule_functions[rule_name](
                        value, rule_config.get('params', {})
                    )
                    applied_rules.append(rule_name)
                    if rule_metadata:
                        metadata[rule_name] = rule_metadata
                except Exception as e:
                    logger.warning(f"Error applying rule {rule_name}: {e}")
        
        if return_metadata:
            metadata['applied_rules'] = applied_rules
            metadata['original_value'] = original_value
            metadata['normalized_value'] = value
            return value, metadata
        else:
            return value
    
    def _trim_whitespace(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Trim leading and trailing whitespace, normalize internal whitespace."""
        normalized = normalize_whitespace(value)
        return normalized, {"changed": normalized != value}
    
    def _remove_space_between_number_and_unit(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Remove space between numbers and units."""
        normalized = remove_space_between_number_and_unit(value)
        return normalized, {"changed": normalized != value}
    
    def _normalize_thousands_separators(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Remove thousands separators from numbers."""
        normalized = normalize_thousands_separators(value)
        return normalized, {"changed": normalized != value}
    
    def _extract_parenthetical_comments(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract parenthetical comments and return cleaned value."""
        cleaned_value, comment = extract_parenthetical_comments(value)
        metadata = {"had_comment": comment is not None}
        if comment:
            metadata["extracted_comment"] = comment
        return cleaned_value, metadata
    
    def _case_normalization(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Normalize case based on parameters."""
        case_type = params.get('case', 'lower')  # 'lower', 'upper', 'title', 'preserve'
        
        if case_type == 'lower':
            normalized = value.lower()
        elif case_type == 'upper':
            normalized = value.upper()
        elif case_type == 'title':
            normalized = value.title()
        else:  # preserve
            normalized = value
        
        return normalized, {"changed": normalized != value, "case_type": case_type}
    
    def _remove_extra_punctuation(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Remove extra punctuation characters."""
        # Remove multiple consecutive punctuation characters
        normalized = re.sub(r'([^\w\s])\1+', r'\1', value)
        return normalized, {"changed": normalized != value}
    
    def _normalize_units(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Normalize unit symbols and abbreviations."""
        unit_mappings = params.get('unit_mappings', {})
        normalized = value
        
        for old_unit, new_unit in unit_mappings.items():
            # Replace unit with word boundaries
            pattern = r'\b' + re.escape(old_unit) + r'\b'
            normalized = re.sub(pattern, new_unit, normalized, flags=re.IGNORECASE)
        
        return normalized, {"changed": normalized != value}
    
    def _remove_leading_zeros(self, value: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Remove leading zeros from numbers."""
        # Pattern: leading zeros followed by digits
        normalized = re.sub(r'\b0+(\d+)', r'\1', value)
        return normalized, {"changed": normalized != value}
    
    def generate_normalization_rules(self, key_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate normalization rules based on key profiles.
        
        Args:
            key_profiles: Dictionary of key profiles from PatternProfiler
        
        Returns:
            Dictionary of normalization rules per key
        """
        normalization_rules = {}
        
        for composite_key, profile in key_profiles.items():
            rules = []
            
            # Get hints from profile analysis
            hints = self._get_normalization_hints(profile)
            
            # Convert hints to rule configurations
            for hint in hints:
                rule_config = self._create_rule_config(hint, profile)
                if rule_config:
                    rules.append(rule_config)
            
            # Add confidence and priority
            normalization_rules[composite_key] = {
                'rules': rules,
                'confidence': self._calculate_rule_confidence(profile, rules),
                'last_updated': pd.Timestamp.now().isoformat()
            }
        
        logger.info(f"Generated normalization rules for {len(normalization_rules)} keys")
        return normalization_rules
    
    def _get_normalization_hints(self, profile: Dict[str, Any]) -> List[str]:
        """Get normalization hints from a key profile."""
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
        
        # Comma handling
        if patterns.get('pct_commas', 0) > 10:
            hints.append('normalize_thousands_separators')
        
        # Unit spacing
        if profile.get('is_numeric_unit', False):
            hints.append('remove_space_between_number_and_unit')
        
        # Case normalization
        if char_types.get('pct_letters', 0) > 80:
            hints.append('case_normalization')
        
        # Extra punctuation
        if patterns.get('pct_special', 0) > 20:
            hints.append('remove_extra_punctuation')
        
        return hints
    
    def _create_rule_config(self, hint: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create rule configuration from hint and profile."""
        base_configs = {
            'trim_whitespace': {
                'name': 'trim_whitespace',
                'priority': 1,
                'params': {}
            },
            'extract_parenthetical_comments': {
                'name': 'extract_parenthetical_comments',
                'priority': 2,
                'params': {}
            },
            'normalize_thousands_separators': {
                'name': 'normalize_thousands_separators',
                'priority': 3,
                'params': {}
            },
            'remove_space_between_number_and_unit': {
                'name': 'remove_space_between_number_and_unit',
                'priority': 4,
                'params': {}
            },
            'case_normalization': {
                'name': 'case_normalization',
                'priority': 5,
                'params': {'case': 'lower'}
            },
            'remove_extra_punctuation': {
                'name': 'remove_extra_punctuation',
                'priority': 6,
                'params': {}
            }
        }
        
        return base_configs.get(hint)
    
    def _calculate_rule_confidence(self, profile: Dict[str, Any], rules: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for normalization rules."""
        if not rules:
            return 0.0
        
        # Base confidence on how well the rules match the data characteristics
        value_analysis = profile.get('value_analysis', {})
        patterns = value_analysis.get('patterns', {})
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear patterns
        if patterns.get('pct_whitespace', 0) > 20:
            confidence += 0.1
        if patterns.get('pct_parentheses', 0) > 10:
            confidence += 0.1
        if profile.get('is_numeric_unit', False):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def validate_normalization_rules(self, test_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate normalization rules against test data.
        
        Args:
            test_data: Dictionary mapping composite keys to lists of test values
        
        Returns:
            Validation results
        """
        results = {}
        
        for composite_key, test_values in test_data.items():
            if composite_key not in self.normalization_rules:
                results[composite_key] = {"error": "No rules found for key"}
                continue
            
            normalized_values = []
            changes_made = 0
            
            for value in test_values:
                normalized, metadata = self.normalize_value(value, composite_key, return_metadata=True)
                normalized_values.append(normalized)
                if metadata.get('applied_rules'):
                    changes_made += 1
            
            results[composite_key] = {
                "total_values": len(test_values),
                "values_changed": changes_made,
                "change_rate": changes_made / len(test_values) if test_values else 0,
                "normalized_values": normalized_values[:5]  # Show first 5 examples
            }
        
        return results