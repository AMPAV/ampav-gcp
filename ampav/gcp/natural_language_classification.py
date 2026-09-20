"""Native Google Cloud Natural Language content classification."""

from google.cloud import language_v1


class GcpNaturalLanguageClassification:
    """Classify direct text and return Google's native category response.

    The model and taxonomy version retain Google's V1/V2 distinction. Text
    preparation, chunking, persistence, and AMPAV schema conversion remain
    client responsibilities.
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
        model: str = "v2",
        content_categories_version: str = "v2",
    ) -> language_v1.ClassifyTextRequest:
        """Build a native classify-text request for a documented model."""
        if not text.strip():
            raise ValueError("text must not be empty")
        if model not in {"v1", "v2"}:
            raise ValueError("model must be 'v1' or 'v2'")
        if content_categories_version not in {"v1", "v2"}:
            raise ValueError("content_categories_version must be 'v1' or 'v2'")

        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
        )
        if language:
            document.language = language

        if model == "v1":
            options = language_v1.ClassificationModelOptions(
                v1_model=language_v1.ClassificationModelOptions.V1Model()
            )
        else:
            category_version = (
                language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V1
                if content_categories_version == "v1"
                else language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
            )
            options = language_v1.ClassificationModelOptions(
                v2_model=language_v1.ClassificationModelOptions.V2Model(
                    content_categories_version=category_version
                )
            )
        return language_v1.ClassifyTextRequest(
            document=document,
            classification_model_options=options,
        )

    def process(
        self,
        text: str,
        *,
        language: str | None = None,
        model: str = "v2",
        content_categories_version: str = "v2",
        timeout: float | None = None,
    ) -> language_v1.ClassifyTextResponse:
        """Classify text synchronously and return the native response."""
        request = self.build_request(
            text,
            language=language,
            model=model,
            content_categories_version=content_categories_version,
        )
        return self.client.classify_text(request=request, timeout=timeout)
