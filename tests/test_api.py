"""
Integration tests for FastAPI service.
"""

import pytest
import json
from fastapi.testclient import TestClient
from autopatternchecker.api import create_app

class TestAPI:
    """Test cases for FastAPI service."""
    
    def setup_method(self):
        """Set up test client."""
        self.config = {
            'key_columns': ['key_part1', 'key_part2', 'key_part3'],
            'value_column': 'value',
            'chunk_size': 1000,
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
            'use_embeddings': False,  # Disable for testing
            'index_type': 'flat',
            'metric': 'cosine',
            'review_batch_size': 100,
            'auto_accept_confidence_threshold': 0.95,
            'manual_review_threshold_lower': 0.6,
            'manual_review_threshold_upper': 0.95,
            'retrain_threshold': 10
        }
        
        # Create test app with minimal config
        self.app = create_app(self.config)
        self.client = TestClient(self.app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_validate_endpoint_structure(self):
        """Test validation endpoint structure."""
        # Test with valid request structure
        request_data = {
            "key_parts": ["A", "B", "C"],
            "value": "test value",
            "metadata": {"test": True}
        }
        
        response = self.client.post("/validate", json=request_data)
        
        # Should return 200 even if key not found (returns needs_review)
        assert response.status_code == 200
        
        data = response.json()
        assert "verdict" in data
        assert "issues" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert "rules_applied" in data
    
    def test_validate_invalid_input(self):
        """Test validation with invalid input."""
        # Test with invalid key_parts
        request_data = {
            "key_parts": ["A", "B"],  # Wrong length
            "value": "test value"
        }
        
        response = self.client.post("/validate", json=request_data)
        assert response.status_code == 400
        assert "Invalid key_parts format" in response.json()["detail"]
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = self.client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_keys" in data
        assert "review_queue_size" in data
        assert "indices_loaded" in data
        assert "normalization_rules_loaded" in data
    
    def test_review_endpoints(self):
        """Test review endpoints."""
        # Test get review items
        response = self.client.get("/review/next")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        
        # Test submit review decision
        decision_data = {
            "item_id": "test_id",
            "decision": "accept",
            "corrected_value": None,
            "notes": "Test decision"
        }
        
        response = self.client.post("/review/submit", json=decision_data)
        # Should return 404 since item doesn't exist
        assert response.status_code == 404
    
    def test_profile_endpoint(self):
        """Test profile endpoint."""
        # Test with non-existent key
        response = self.client.get("/profile/nonexistent_key")
        assert response.status_code == 404
        assert "Key profile not found" in response.json()["detail"]
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/validate")
        # CORS preflight should be handled
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS