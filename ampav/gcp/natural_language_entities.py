"""Native Google Cloud Natural Language entity analysis."""

from google.cloud import language_v1


class GcpNaturalLanguageEntities:
    """Analyze direct text and return Google's native entity response.

    Authentication and quota-project selection use Application Default
    Credentials through the Google client. Text preparation, chunking,
    persistence, and AMPAV schema conversion remain client responsibilities.
    """

    def __init__(
        self,
        *,
        client: language_v1.LanguageServiceClient | None = None,
    ) -> None:
        self.client = client or language_v1.LanguageServiceClient()

    def build_request(
        self,
        text: str,
        *,
        language: str | None = None,
        encoding_type: language_v1.EncodingType = language_v1.EncodingType.UTF8,
    ) -> language_v1.AnalyzeEntitiesRequest:
        """Build a native entity-analysis request for plain text."""
        if not text.strip():
            raise ValueError("text must not be empty")

        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
        )
        if language:
            document.language = language
        return language_v1.AnalyzeEntitiesRequest(
            document=document,
            encoding_type=encoding_type,
        )

    def process(
        self,
        text: str,
        *,
        language: str | None = None,
        encoding_type: language_v1.EncodingType = language_v1.EncodingType.UTF8,
        timeout: float | None = None,
    ) -> language_v1.AnalyzeEntitiesResponse:
        """Analyze text synchronously and return the native Google response."""
        request = self.build_request(
            text,
            language=language,
            encoding_type=encoding_type,
        )
        return self.client.analyze_entities(request=request, timeout=timeout)
