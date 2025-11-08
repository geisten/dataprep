"""Text-Verarbeitungsmodule"""

from .html_extractor import HTMLExtractor
from .cleaner import TextCleaner
from .deduplicator import ExactDeduplicator, FuzzyDeduplicator, SoftDeduplicator

__all__ = [
    "HTMLExtractor",
    "TextCleaner",
    "ExactDeduplicator",
    "FuzzyDeduplicator",
    "SoftDeduplicator",
]
