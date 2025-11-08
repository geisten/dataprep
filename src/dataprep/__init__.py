"""
LLM Pre-Training Data Preparation Pipeline

Moderne Datenaufbereitungs-Pipeline basierend auf Phi-3/4 und neuesten Best Practices.
"""

__version__ = "0.1.0"

from .pipeline import (
    TextPipeline,
    MultimodalPipeline,
    CompletePipeline,
)

from .text.html_extractor import HTMLExtractor
from .text.cleaner import TextCleaner
from .text.deduplicator import (
    ExactDeduplicator,
    FuzzyDeduplicator,
    SoftDeduplicator,
)

from .quality.filter import QualityFilter
from .quality.scorer import QualityScorer

__all__ = [
    "__version__",
    "TextPipeline",
    "MultimodalPipeline",
    "CompletePipeline",
    "HTMLExtractor",
    "TextCleaner",
    "ExactDeduplicator",
    "FuzzyDeduplicator",
    "SoftDeduplicator",
    "QualityFilter",
    "QualityScorer",
]
