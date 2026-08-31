"""GCP tools for AMPAV."""

from .natural_language_entities import GcpNaturalLanguageEntities
from .speech_to_text import GcpSpeechToTextBatch

__version__ = "0.0.1"


__all__ = [
    "GcpNaturalLanguageEntities",
    "GcpSpeechToTextBatch",
    "__version__",
]
