"""GCP tools for AMPAV."""

from .natural_language_entities import GcpNaturalLanguageEntities
from .natural_language_classification import GcpNaturalLanguageClassification
from .speech_to_text import GcpSpeechToTextBatch

__version__ = "0.0.1"


__all__ = [
    "GcpNaturalLanguageEntities",
    "GcpNaturalLanguageClassification",
    "GcpSpeechToTextBatch",
    "__version__",
]
