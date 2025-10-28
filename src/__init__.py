"""
Research Paper Authenticity Detection System
"""

__version__ = "1.0.0"
__author__ = "Research Integrity Team"

from .document_processor import DocumentProcessor
from .ai_detector import AIContentDetector
from .plagiarism_detector import SemanticPlagiarismDetector
from .explainability import ExplainabilityEngine, AuthenticityScore

__all__ = [
    'DocumentProcessor',
    'AIContentDetector',
    'SemanticPlagiarismDetector',
    'ExplainabilityEngine',
    'AuthenticityScore'
]