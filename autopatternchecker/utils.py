"""
Utility functions for AutoPatternChecker.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_composite_key(row: pd.Series, key_columns: List[str]) -> str:
    """Create composite key from specified columns."""
    return "||".join(str(row[col]) for col in key_columns)

def generate_signature(value: str) -> str:
    """
    Generate character signature for a value.
    Maps characters to tokens: digit->D, letter->L, whitespace->_, other->O
    Compresses consecutive repeats.
    """
    if pd.isna(value) or value == "":
        return ""
    
    value = str(value)
    signature = []
    current_char = None
    count = 0
    
    for char in value:
        if char.isdigit():
            char_type = 'D'
        elif char.isalpha():
            char_type = 'L'
        elif char.isspace():
            char_type = '_'
        else:
            char_type = 'O'
        
        if char_type == current_char:
            count += 1
        else:
            if current_char is not None:
                if count > 1:
                    signature.append(f"{current_char}{count}")
                else:
                    signature.append(current_char)
            current_char = char_type
            count = 1
    
    # Add the last character
    if current_char is not None:
        if count > 1:
            signature.append(f"{current_char}{count}")
        else:
            signature.append(current_char)
    
    return "".join(signature)

def signature_to_regex(signature: str) -> str:
    """
    Convert signature to candidate regex pattern.
    This is a heuristic mapping that can be refined.
    """
    if not signature:
        return ".*"
    
    # Basic mapping rules
    pattern_parts = []
    i = 0
    while i < len(signature):
        char = signature[i]
        if char == 'D':
            # Check if followed by number (compressed)
            if i + 1 < len(signature) and signature[i + 1].isdigit():
                # Extract the count
                count_str = ""
                j = i + 1
                while j < len(signature) and signature[j].isdigit():
                    count_str += signature[j]
                    j += 1
                count = int(count_str)
                pattern_parts.append(f"\\d{{{count}}}")
                i = j
            else:
                pattern_parts.append("\\d+")
                i += 1
        elif char == 'L':
            # Check if followed by number (compressed)
            if i + 1 < len(signature) and signature[i + 1].isdigit():
                count_str = ""
                j = i + 1
                while j < len(signature) and signature[j].isdigit():
                    count_str += signature[j]
                    j += 1
                count = int(count_str)
                pattern_parts.append(f"[A-Za-z\\u0600-\\u06FF]{{{count}}}")
                i = j
            else:
                pattern_parts.append("[A-Za-z\\u0600-\\u06FF]+")
                i += 1
        elif char == '_':
            pattern_parts.append("\\s*")
            i += 1
        elif char == 'O':
            pattern_parts.append("[^\\w\\s]+")
            i += 1
        else:
            i += 1
    
    return "".join(pattern_parts)

def normalize_whitespace(value: str) -> str:
    """Normalize whitespace in a string."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())

def remove_space_between_number_and_unit(value: str) -> str:
    """Remove space between numbers and common units."""
    if pd.isna(value):
        return ""
    
    value = str(value)
    # Common units pattern
    units = ['V', 'A', 'W', 'Hz', 'MHz', 'GHz', 'kHz', 'Ω', 'Ohm', 'F', 'H', 'C', 'K', 'J']
    
    for unit in units:
        # Pattern: number + space + unit
        pattern = r'(\d+)\s+(' + re.escape(unit) + r')(?=\s|$)'
        value = re.sub(pattern, r'\1\2', value)
    
    return value

def normalize_thousands_separators(value: str) -> str:
    """Remove thousands separators (commas) from numbers."""
    if pd.isna(value):
        return ""
    
    value = str(value)
    # Pattern: digit, comma, digit, digit, digit (end of number or followed by non-digit)
    pattern = r'(\d),(\d{3})(?=\D|$)'
    while re.search(pattern, value):
        value = re.sub(pattern, r'\1\2', value)
    
    return value

def extract_parenthetical_comments(value: str) -> Tuple[str, Optional[str]]:
    """Extract parenthetical comments from value."""
    if pd.isna(value):
        return "", None
    
    value = str(value)
    # Find content in parentheses
    match = re.search(r'\(([^)]+)\)', value)
    if match:
        comment = match.group(1)
        # Remove the parentheses and content
        cleaned_value = re.sub(r'\s*\([^)]+\)\s*', ' ', value).strip()
        return cleaned_value, comment
    
    return value, None

def calculate_similarity_stats(similarities: List[float]) -> Dict[str, float]:
    """Calculate statistics for similarity scores."""
    if not similarities:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    
    similarities = np.array(similarities)
    return {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "median": float(np.median(similarities))
    }

def validate_composite_key_format(key_parts: List[str], expected_length: int = 3) -> bool:
    """Validate that composite key parts are properly formatted."""
    if len(key_parts) != expected_length:
        return False
    
    for part in key_parts:
        if not isinstance(part, str) or part.strip() == "":
            return False
    
    return True

def create_metadata_entry(
    value: str,
    composite_key: str,
    verdict: str,
    confidence: float,
    issues: List[str],
    suggested_fix: Optional[str] = None,
    nearest_examples: Optional[List[str]] = None,
    rules_applied: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a standardized metadata entry for validation results."""
    return {
        "timestamp": datetime.now().isoformat(),
        "composite_key": composite_key,
        "original_value": value,
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "suggested_fix": suggested_fix,
        "nearest_examples": nearest_examples or [],
        "rules_applied": rules_applied or []
    }

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """Split DataFrame into chunks for memory-efficient processing."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size].copy())
    return chunks

def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects to JSON, handling numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj