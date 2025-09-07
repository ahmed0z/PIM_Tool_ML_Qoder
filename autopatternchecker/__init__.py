"""
AutoPatternChecker: Automated system that learns per-composite-key formats from CSV data.

This package provides tools for:
- Data ingestion and profiling
- Pattern learning and normalization
- Clustering and embedding-based duplicate detection
- Runtime validation API
- Active learning for continuous improvement
"""

__version__ = "1.0.0"
__author__ = "AutoPatternChecker Team"

from .ingest import DataIngester
from .profiling import PatternProfiler
from .normalize import NormalizationEngine
from .clustering import ClusterAnalyzer
from .embeddings import EmbeddingGenerator
from .indexing import FAISSIndexer
from .api import create_app
from .training import ModelTrainer
from .active_learning import ActiveLearningManager

__all__ = [
    "DataIngester",
    "PatternProfiler",
    "NormalizationEngine",
    "ClusterAnalyzer",
    "EmbeddingGenerator",
    "FAISSIndexer",
    "create_app",
    "ModelTrainer",
    "ActiveLearningManager"
]