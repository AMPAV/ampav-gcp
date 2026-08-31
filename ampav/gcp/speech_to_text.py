"""Native Google Cloud Speech-to-Text V2 batch recognition."""

from collections.abc import Sequence
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


class GcpSpeechToTextBatch:
    """Submit Chirp 3 batch requests and return native Google responses.

    The wrapper accepts an existing GCS URI and uses Application Default
    Credentials through the Google client. Uploads, persistence, cleanup, and
    AMPAV transcript conversion remain client responsibilities.
    """

    def __init__(
        self,
        project_id: str,
        *,
        location: str = "us",
        model: str = "chirp_3",
        client: SpeechClient | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id must not be empty")
        if not location:
            raise ValueError("location must not be empty")
        if not model:
            raise ValueError("model must not be empty")

        self.project_id = project_id
        self.location = location
        self.model = model
        self.client = client or SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com",
            )
        )

    @property
    def recognizer(self) -> str:
        """Return the implicit recognizer resource used by this wrapper."""
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            "recognizers/_"
        )

    def build_request(
        self,
        gcs_uri: str,
        *,
        language_codes: Sequence[str] = ("en-US",),
        enable_word_time_offsets: bool = True,
        enable_diarization: bool = True,
    ) -> cloud_speech.BatchRecognizeRequest:
        """Build a one-file native BatchRecognize request with inline output."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError("gcs_uri must start with gs://")
        if not language_codes or any(not code for code in language_codes):
            raise ValueError("language_codes must contain at least one value")

        features = cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=enable_word_time_offsets,
        )
        if enable_diarization:
            features.diarization_config = cloud_speech.SpeakerDiarizationConfig()

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=list(language_codes),
            model=self.model,
            features=features,
        )
        return cloud_speech.BatchRecognizeRequest(
            recognizer=self.recognizer,
            config=config,
            files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                inline_response_config=cloud_speech.InlineOutputConfig(),
            ),
        )

    def submit(
        self,
        gcs_uri: str,
        *,
        language_codes: Sequence[str] = ("en-US",),
        enable_word_time_offsets: bool = True,
        enable_diarization: bool = True,
    ) -> Any:
        """Submit a request and return Google's native long-running operation."""
        request = self.build_request(
            gcs_uri,
            language_codes=language_codes,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_diarization=enable_diarization,
        )
        return self.client.batch_recognize(request=request)

    def process(
        self,
        gcs_uri: str,
        *,
        language_codes: Sequence[str] = ("en-US",),
        enable_word_time_offsets: bool = True,
        enable_diarization: bool = True,
        timeout: float | None = None,
    ) -> cloud_speech.BatchRecognizeResponse:
        """Submit and wait for a native BatchRecognize response."""
        operation = self.submit(
            gcs_uri,
            language_codes=language_codes,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_diarization=enable_diarization,
        )
        return operation.result(timeout=timeout)
