"""
FastAPI service for AutoPatternChecker runtime validation.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import logging
import json
from datetime import datetime
import asyncio
from .utils import create_composite_key, validate_composite_key_format, create_metadata_entry
from .normalize import NormalizationEngine
from .indexing import FAISSIndexer
from .embeddings import EmbeddingGenerator
from .profiling import PatternProfiler
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Pydantic models
class ValidationRequest(BaseModel):
    key_parts: List[str] = Field(..., description="List of key parts to form composite key")
    value: str = Field(..., description="Value to validate")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class ValidationResponse(BaseModel):
    verdict: str = Field(..., description="Validation verdict")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggested_fix: Optional[str] = Field(None, description="Suggested corrected value")
    nearest_examples: List[str] = Field(default_factory=list, description="Nearest similar examples")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    rules_applied: List[str] = Field(default_factory=list, description="Normalization rules applied")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ReviewItem(BaseModel):
    id: str
    composite_key: str
    original_value: str
    suggested_fix: Optional[str]
    nearest_examples: List[str]
    confidence: float
    timestamp: str
    issues: List[str]

class ReviewDecision(BaseModel):
    item_id: str
    decision: str = Field(..., description="accept, edit, or reject")
    corrected_value: Optional[str] = Field(None, description="Corrected value if decision is edit")
    notes: Optional[str] = Field(None, description="Optional notes")

class ProfileResponse(BaseModel):
    composite_key: str
    count: int
    unique_values: int
    unique_ratio: float
    is_free_text: bool
    is_numeric_unit: bool
    top_signatures: List[Dict[str, Any]]
    candidate_regex: Optional[str]
    normalization_rules: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    similarity_threshold: float

# Global variables for loaded models
normalization_engine = None
faiss_indexer = None
embedding_generator = None
pattern_profiler = None
key_profiles = {}
review_queue = []
review_counter = 0

def create_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="AutoPatternChecker API",
        description="Automated pattern validation and correction service",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize global components
    global normalization_engine, faiss_indexer, embedding_generator, pattern_profiler
    
    normalization_engine = NormalizationEngine(config)
    faiss_indexer = FAISSIndexer(config)
    embedding_generator = EmbeddingGenerator(config)
    pattern_profiler = PatternProfiler(config)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the service on startup."""
        try:
            # Load normalization rules
            rules_path = config.get('normalization_rules_path')
            if rules_path:
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
                normalization_engine.load_normalization_rules(rules)
            
            # Load FAISS indices
            indices_path = config.get('faiss_indices_path')
            if indices_path:
                faiss_indexer.load_indices(indices_path)
            
            # Load key profiles
            profiles_path = config.get('key_profiles_path')
            if profiles_path:
                with open(profiles_path, 'r') as f:
                    global key_profiles
                    key_profiles = json.load(f)
            
            logger.info("AutoPatternChecker API initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            raise
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    
    @app.post("/validate", response_model=ValidationResponse)
    async def validate_value(request: ValidationRequest, background_tasks: BackgroundTasks):
        """Validate a value for a given composite key."""
        start_time = datetime.now()
        
        try:
            # Validate input
            if not validate_composite_key_format(request.key_parts, expected_length=3):
                raise HTTPException(status_code=400, detail="Invalid key_parts format")
            
            # Create composite key
            composite_key = create_composite_key(
                pd.Series(request.key_parts), 
                ['key_part1', 'key_part2', 'key_part3']
            )
            
            # Check if we have a profile for this key
            if composite_key not in key_profiles:
                return ValidationResponse(
                    verdict="needs_review",
                    issues=["unknown_key"],
                    confidence=0.0,
                    processing_time_ms=0.0
                )
            
            # Get key profile
            profile = key_profiles[composite_key]
            
            # Normalize the value
            normalized_value, norm_metadata = normalization_engine.normalize_value(
                request.value, composite_key, return_metadata=True
            )
            
            # Check format against regex if available
            format_match = True
            format_issues = []
            candidate_regex = profile.get('candidate_regex')
            
            if candidate_regex:
                import re
                if not re.match(candidate_regex, normalized_value):
                    format_match = False
                    format_issues.append("format_mismatch")
            
            # Check for semantic duplicates using embeddings
            duplicate_info = await check_semantic_duplicates(
                composite_key, normalized_value, profile
            )
            
            # Determine verdict
            verdict, issues, suggested_fix, confidence = determine_verdict(
                format_match, format_issues, duplicate_info, profile
            )
            
            # Get nearest examples
            nearest_examples = duplicate_info.get('nearest_examples', [])
            
            # Create metadata entry for logging
            metadata_entry = create_metadata_entry(
                request.value, composite_key, verdict, confidence, issues,
                suggested_fix, nearest_examples, norm_metadata.get('applied_rules', [])
            )
            
            # Add to review queue if needed
            if verdict == "needs_review":
                background_tasks.add_task(add_to_review_queue, metadata_entry)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ValidationResponse(
                verdict=verdict,
                issues=issues,
                suggested_fix=suggested_fix,
                nearest_examples=nearest_examples,
                confidence=confidence,
                rules_applied=norm_metadata.get('applied_rules', []),
                processing_time_ms=round(processing_time, 2)
            )
            
        except Exception as e:
            logger.error(f"Error validating value: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/profile/{composite_key}", response_model=ProfileResponse)
    async def get_key_profile(composite_key: str):
        """Get profile information for a specific composite key."""
        if composite_key not in key_profiles:
            raise HTTPException(status_code=404, detail="Key profile not found")
        
        profile = key_profiles[composite_key]
        
        # Get normalization rules for this key
        key_rules = normalization_engine.normalization_rules.get(composite_key, {}).get('rules', [])
        
        return ProfileResponse(
            composite_key=composite_key,
            count=profile.get('count', 0),
            unique_values=profile.get('unique_values', 0),
            unique_ratio=profile.get('unique_ratio', 0.0),
            is_free_text=profile.get('is_free_text', False),
            is_numeric_unit=profile.get('is_numeric_unit', False),
            top_signatures=profile.get('top_signatures', []),
            candidate_regex=profile.get('candidate_regex'),
            normalization_rules=key_rules,
            clusters=profile.get('clusters', []),
            similarity_threshold=profile.get('similarity_threshold', 0.82)
        )
    
    @app.get("/review/next", response_model=List[ReviewItem])
    async def get_review_items(limit: int = 10):
        """Get next items for human review."""
        global review_queue
        
        items = review_queue[:limit]
        return items
    
    @app.post("/review/submit")
    async def submit_review_decision(decision: ReviewDecision, background_tasks: BackgroundTasks):
        """Submit a review decision for a flagged item."""
        global review_queue
        
        # Find and remove the item from review queue
        item_found = False
        for i, item in enumerate(review_queue):
            if item['id'] == decision.item_id:
                del review_queue[i]
                item_found = True
                break
        
        if not item_found:
            raise HTTPException(status_code=404, detail="Review item not found")
        
        # Process the decision
        background_tasks.add_task(process_review_decision, decision)
        
        return {"status": "accepted", "message": "Review decision processed"}
    
    @app.get("/stats")
    async def get_service_stats():
        """Get service statistics."""
        return {
            "total_keys": len(key_profiles),
            "review_queue_size": len(review_queue),
            "indices_loaded": len(faiss_indexer.indices),
            "normalization_rules_loaded": len(normalization_engine.normalization_rules)
        }
    
    return app

async def check_semantic_duplicates(composite_key: str, value: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Check for semantic duplicates using embeddings and FAISS index."""
    try:
        # Generate embedding for the value
        embedding = embedding_generator.generate_embeddings([value])
        if len(embedding) == 0:
            return {"is_duplicate": False, "similarity": 0.0, "nearest_examples": []}
        
        # Search for similar values
        similar_results = faiss_indexer.search_similar(
            composite_key, embedding[0], 
            top_k=5, 
            threshold=profile.get('similarity_threshold', 0.82)
        )
        
        if similar_results:
            best_match = similar_results[0]
            return {
                "is_duplicate": True,
                "similarity": best_match['similarity'],
                "nearest_examples": [r['value'] for r in similar_results[:3]]
            }
        else:
            return {"is_duplicate": False, "similarity": 0.0, "nearest_examples": []}
    
    except Exception as e:
        logger.warning(f"Error checking semantic duplicates: {e}")
        return {"is_duplicate": False, "similarity": 0.0, "nearest_examples": []}

def determine_verdict(format_match: bool, format_issues: List[str], 
                     duplicate_info: Dict[str, Any], profile: Dict[str, Any]) -> tuple[str, List[str], Optional[str], float]:
    """Determine the validation verdict based on all checks."""
    
    issues = format_issues.copy()
    suggested_fix = None
    confidence = 1.0
    
    # Check for semantic duplicates
    if duplicate_info.get('is_duplicate', False):
        similarity = duplicate_info.get('similarity', 0.0)
        if similarity > 0.95:
            return "duplicate", ["near_duplicate"], None, similarity
        elif similarity > 0.85:
            issues.append("near_duplicate")
            confidence *= 0.8
    
    # Check format
    if not format_match:
        issues.append("format_mismatch")
        confidence *= 0.6
        
        # Try to suggest a fix
        if duplicate_info.get('nearest_examples'):
            suggested_fix = duplicate_info['nearest_examples'][0]
    
    # Determine verdict
    if not issues:
        verdict = "accepted"
    elif "format_mismatch" in issues and suggested_fix:
        verdict = "reformatted"
    elif "near_duplicate" in issues:
        verdict = "duplicate"
    elif confidence < 0.6:
        verdict = "needs_review"
    else:
        verdict = "accepted"
    
    return verdict, issues, suggested_fix, confidence

def add_to_review_queue(metadata_entry: Dict[str, Any]):
    """Add an item to the review queue."""
    global review_queue, review_counter
    
    review_counter += 1
    review_item = ReviewItem(
        id=f"review_{review_counter}",
        composite_key=metadata_entry['composite_key'],
        original_value=metadata_entry['original_value'],
        suggested_fix=metadata_entry.get('suggested_fix'),
        nearest_examples=metadata_entry.get('nearest_examples', []),
        confidence=metadata_entry['confidence'],
        timestamp=metadata_entry['timestamp'],
        issues=metadata_entry['issues']
    )
    
    review_queue.append(review_item.dict())
    logger.info(f"Added item to review queue: {review_item.id}")

def process_review_decision(decision: ReviewDecision):
    """Process a human review decision."""
    logger.info(f"Processing review decision for {decision.item_id}: {decision.decision}")
    
    # In a full implementation, this would:
    # 1. Update normalization rules if needed
    # 2. Add the example to training data
    # 3. Trigger model updates if necessary
    # 4. Log the decision for analysis
    
    # For now, just log the decision
    decision_log = {
        "item_id": decision.item_id,
        "decision": decision.decision,
        "corrected_value": decision.corrected_value,
        "notes": decision.notes,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Review decision logged: {decision_log}")