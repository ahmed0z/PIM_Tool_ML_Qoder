"""
Active learning module for AutoPatternChecker.
Handles human-in-the-loop improvements and continuous learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from .utils import save_json, load_json

logger = logging.getLogger(__name__)

class ActiveLearningManager:
    """Manages active learning and human feedback integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.review_batch_size = config.get('review_batch_size', 100)
        self.auto_accept_threshold = config.get('auto_accept_confidence_threshold', 0.95)
        self.manual_review_lower = config.get('manual_review_threshold_lower', 0.6)
        self.manual_review_upper = config.get('manual_review_threshold_upper', 0.95)
        self.retrain_threshold = config.get('retrain_threshold', 10)  # Number of changes before retrain
        
        # Storage for review data
        self.review_queue = []
        self.review_history = []
        self.labeled_examples = {}
        self.rule_updates = {}
        
    def add_to_review_queue(self, validation_result: Dict[str, Any]) -> str:
        """
        Add a validation result to the review queue.
        
        Args:
            validation_result: Result from validation API
        
        Returns:
            Review item ID
        """
        review_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.review_queue)}"
        
        review_item = {
            'id': review_id,
            'composite_key': validation_result['composite_key'],
            'original_value': validation_result['original_value'],
            'verdict': validation_result['verdict'],
            'confidence': validation_result['confidence'],
            'issues': validation_result['issues'],
            'suggested_fix': validation_result.get('suggested_fix'),
            'nearest_examples': validation_result.get('nearest_examples', []),
            'rules_applied': validation_result.get('rules_applied', []),
            'timestamp': validation_result['timestamp'],
            'status': 'pending',
            'human_decision': None,
            'human_notes': None
        }
        
        self.review_queue.append(review_item)
        logger.info(f"Added item to review queue: {review_id}")
        
        return review_id
    
    def get_review_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a batch of items for human review.
        
        Args:
            batch_size: Optional batch size override
        
        Returns:
            List of review items
        """
        size = batch_size or self.review_batch_size
        return self.review_queue[:size]
    
    def submit_review_decision(self, review_id: str, decision: str, 
                              corrected_value: Optional[str] = None,
                              notes: Optional[str] = None) -> bool:
        """
        Submit a human review decision.
        
        Args:
            review_id: ID of the review item
            decision: Human decision (accept, edit, reject)
            corrected_value: Corrected value if decision is 'edit'
            notes: Optional notes
        
        Returns:
            True if successful
        """
        # Find the review item
        review_item = None
        for item in self.review_queue:
            if item['id'] == review_id:
                review_item = item
                break
        
        if not review_item:
            logger.error(f"Review item not found: {review_id}")
            return False
        
        # Update the item
        review_item['status'] = 'reviewed'
        review_item['human_decision'] = decision
        review_item['human_notes'] = notes
        review_item['reviewed_at'] = datetime.now().isoformat()
        
        if decision == 'edit' and corrected_value:
            review_item['corrected_value'] = corrected_value
        
        # Move to history
        self.review_history.append(review_item)
        self.review_queue.remove(review_item)
        
        # Process the decision
        self._process_review_decision(review_item)
        
        logger.info(f"Processed review decision for {review_id}: {decision}")
        return True
    
    def _process_review_decision(self, review_item: Dict[str, Any]) -> None:
        """Process a human review decision and update models accordingly."""
        composite_key = review_item['composite_key']
        decision = review_item['human_decision']
        
        # Add to labeled examples
        if composite_key not in self.labeled_examples:
            self.labeled_examples[composite_key] = []
        
        labeled_example = {
            'original_value': review_item['original_value'],
            'human_decision': decision,
            'corrected_value': review_item.get('corrected_value'),
            'confidence': review_item['confidence'],
            'timestamp': review_item['reviewed_at']
        }
        
        self.labeled_examples[composite_key].append(labeled_example)
        
        # Update normalization rules if needed
        if decision in ['accept', 'edit']:
            self._update_normalization_rules(composite_key, review_item)
        
        # Check if retraining is needed
        if self._should_retrain(composite_key):
            self._schedule_retraining(composite_key)
    
    def _update_normalization_rules(self, composite_key: str, review_item: Dict[str, Any]) -> None:
        """Update normalization rules based on human feedback."""
        if composite_key not in self.rule_updates:
            self.rule_updates[composite_key] = []
        
        rule_update = {
            'timestamp': review_item['reviewed_at'],
            'original_value': review_item['original_value'],
            'corrected_value': review_item.get('corrected_value'),
            'decision': review_item['human_decision'],
            'rules_applied': review_item.get('rules_applied', [])
        }
        
        self.rule_updates[composite_key].append(rule_update)
        
        # Analyze patterns in corrections to suggest new rules
        self._analyze_correction_patterns(composite_key)
    
    def _analyze_correction_patterns(self, composite_key: str) -> None:
        """Analyze patterns in human corrections to suggest new normalization rules."""
        if composite_key not in self.rule_updates:
            return
        
        corrections = [update for update in self.rule_updates[composite_key] 
                      if update['decision'] == 'edit' and update['corrected_value']]
        
        if len(corrections) < 3:  # Need enough examples
            return
        
        # Analyze common patterns
        patterns = self._extract_correction_patterns(corrections)
        
        if patterns:
            logger.info(f"Found correction patterns for {composite_key}: {patterns}")
            # In a full implementation, these patterns would be used to suggest new rules
    
    def _extract_correction_patterns(self, corrections: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from human corrections."""
        patterns = []
        
        # Look for common transformations
        for correction in corrections:
            original = correction['original_value']
            corrected = correction['corrected_value']
            
            # Simple pattern detection
            if len(original) > len(corrected):
                patterns.append('trimming')
            elif original.lower() != corrected.lower():
                patterns.append('case_normalization')
            elif ' ' in original and ' ' not in corrected:
                patterns.append('space_removal')
        
        # Return most common patterns
        from collections import Counter
        pattern_counts = Counter(patterns)
        return [pattern for pattern, count in pattern_counts.most_common(3)]
    
    def _should_retrain(self, composite_key: str) -> bool:
        """Check if retraining is needed for a composite key."""
        if composite_key not in self.labeled_examples:
            return False
        
        recent_examples = [
            ex for ex in self.labeled_examples[composite_key]
            if datetime.fromisoformat(ex['timestamp']) > datetime.now() - timedelta(days=1)
        ]
        
        return len(recent_examples) >= self.retrain_threshold
    
    def _schedule_retraining(self, composite_key: str) -> None:
        """Schedule retraining for a composite key."""
        logger.info(f"Scheduling retraining for {composite_key}")
        # In a full implementation, this would trigger a background retraining job
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics about the review process."""
        total_reviewed = len(self.review_history)
        pending_review = len(self.review_queue)
        
        # Decision distribution
        decisions = [item['human_decision'] for item in self.review_history 
                   if item['human_decision']]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Confidence distribution
        confidences = [item['confidence'] for item in self.review_history]
        confidence_stats = {
            'mean': float(np.mean(confidences)) if confidences else 0.0,
            'std': float(np.std(confidences)) if confidences else 0.0,
            'min': float(np.min(confidences)) if confidences else 0.0,
            'max': float(np.max(confidences)) if confidences else 0.0
        }
        
        return {
            'total_reviewed': total_reviewed,
            'pending_review': pending_review,
            'decision_distribution': decision_counts,
            'confidence_stats': confidence_stats,
            'keys_with_examples': len(self.labeled_examples),
            'keys_with_rule_updates': len(self.rule_updates)
        }
    
    def get_key_learning_progress(self, composite_key: str) -> Dict[str, Any]:
        """Get learning progress for a specific composite key."""
        if composite_key not in self.labeled_examples:
            return {'error': 'No learning data for this key'}
        
        examples = self.labeled_examples[composite_key]
        rule_updates = self.rule_updates.get(composite_key, [])
        
        # Analyze learning trends
        recent_examples = [
            ex for ex in examples
            if datetime.fromisoformat(ex['timestamp']) > datetime.now() - timedelta(days=7)
        ]
        
        # Decision trends
        decisions = [ex['human_decision'] for ex in examples]
        decision_trends = {}
        for decision in set(decisions):
            decision_trends[decision] = decisions.count(decision)
        
        return {
            'composite_key': composite_key,
            'total_examples': len(examples),
            'recent_examples': len(recent_examples),
            'rule_updates': len(rule_updates),
            'decision_trends': decision_trends,
            'last_learning': examples[-1]['timestamp'] if examples else None,
            'learning_rate': len(recent_examples) / 7 if recent_examples else 0.0
        }
    
    def export_learning_data(self, filepath: str) -> None:
        """Export learning data for analysis."""
        data = {
            'review_history': self.review_history,
            'labeled_examples': self.labeled_examples,
            'rule_updates': self.rule_updates,
            'export_timestamp': datetime.now().isoformat()
        }
        
        save_json(data, filepath)
        logger.info(f"Exported learning data to {filepath}")
    
    def import_learning_data(self, filepath: str) -> None:
        """Import learning data from file."""
        data = load_json(filepath)
        
        self.review_history = data.get('review_history', [])
        self.labeled_examples = data.get('labeled_examples', {})
        self.rule_updates = data.get('rule_updates', {})
        
        logger.info(f"Imported learning data from {filepath}")
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old learning data to prevent memory issues."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up review history
        self.review_history = [
            item for item in self.review_history
            if datetime.fromisoformat(item['timestamp']) > cutoff_date
        ]
        
        # Clean up labeled examples
        for key in self.labeled_examples:
            self.labeled_examples[key] = [
                ex for ex in self.labeled_examples[key]
                if datetime.fromisoformat(ex['timestamp']) > cutoff_date
            ]
        
        # Clean up rule updates
        for key in self.rule_updates:
            self.rule_updates[key] = [
                update for update in self.rule_updates[key]
                if datetime.fromisoformat(update['timestamp']) > cutoff_date
            ]
        
        logger.info(f"Cleaned up learning data older than {days_to_keep} days")
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive learning report."""
        stats = self.get_review_statistics()
        
        # Key-specific analysis
        key_analyses = {}
        for key in self.labeled_examples.keys():
            key_analyses[key] = self.get_key_learning_progress(key)
        
        # Learning effectiveness metrics
        total_examples = sum(len(examples) for examples in self.labeled_examples.values())
        avg_confidence = stats['confidence_stats']['mean']
        
        # Rule update effectiveness
        total_rule_updates = sum(len(updates) for updates in self.rule_updates.values())
        
        return {
            'overall_stats': stats,
            'key_analyses': key_analyses,
            'learning_effectiveness': {
                'total_examples': total_examples,
                'avg_confidence': avg_confidence,
                'total_rule_updates': total_rule_updates,
                'keys_learning': len(self.labeled_examples)
            },
            'recommendations': self._generate_recommendations(stats, key_analyses)
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                 key_analyses: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on learning data."""
        recommendations = []
        
        # High review queue
        if stats['pending_review'] > 50:
            recommendations.append("High review queue - consider increasing review capacity")
        
        # Low confidence scores
        if stats['confidence_stats']['mean'] < 0.7:
            recommendations.append("Low average confidence - consider improving pattern detection")
        
        # Keys with many rule updates
        for key, analysis in key_analyses.items():
            if analysis.get('rule_updates', 0) > 20:
                recommendations.append(f"Key {key} has many rule updates - consider pattern refinement")
        
        return recommendations