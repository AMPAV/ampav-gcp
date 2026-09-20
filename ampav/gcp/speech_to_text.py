"""Google Cloud Speech-to-Text V2 batch integration."""

from collections.abc import Sequence
import logging
import time
from typing import Any

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.longrunning import operations_pb2
from google.protobuf.json_format import MessageToDict
from pydantic import Field

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool
from ampav.core.schema import ToolOutput

from .errors import GcpSpeechToTextError
from .speech_to_text_conversion import gcp_speech_to_transcript


class GcpSpeechJobStatus(AsyncJobStatus):
    """Speech operation status with selected GCP details."""

    operation_name: str | None = None
    resource: str | None = None
    input_gcs_uris: list[str] = Field(default_factory=list)


class GcpSpeechToTextBatch(AsyncTool):
    """Run Chirp 3 batch recognition against caller-owned GCS media.

    The lifecycle uses opaque operation names. Completed operation records are
    deleted, but caller-owned GCS inputs are never deleted.
    """

    def __init__(
        self,
        project_id: str,
        *,
        location: str = "us",
        model: str = "chirp_3",
        client: SpeechClient | None = None,
        include_tool_private: bool = False,
        polling_interval: float = 30,
        timeout: float | None = 7200,
    ) -> None:
        if not project_id or not location or not model:
            raise ValueError("project_id, location, and model must not be empty")
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")
        self.project_id = project_id
        self.location = location
        self.model = model
        self.client = client or SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
        )
        self.include_tool_private = include_tool_private
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._job_parameters: dict[str, dict[str, Any]] = {}

    @property
    def recognizer(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/recognizers/_"

    @property
    def operations_parent(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}"

    @property
    def operations_client(self) -> Any:
        return self.client.transport.operations_client

    def build_request(
        self,
        gcs_uri: str,
        *,
        language_codes: Sequence[str] = ("en-US",),
        enable_word_time_offsets: bool = True,
        enable_diarization: bool = True,
    ) -> cloud_speech.BatchRecognizeRequest:
        """Build a one-file native request with inline output."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError("gcs_uri must start with gs://")
        if not language_codes or any(not code for code in language_codes):
            raise ValueError("language_codes must contain at least one value")
        features = cloud_speech.RecognitionFeatures(enable_word_time_offsets=enable_word_time_offsets)
        if enable_diarization:
            features.diarization_config = cloud_speech.SpeakerDiarizationConfig()
        return cloud_speech.BatchRecognizeRequest(
            recognizer=self.recognizer,
            config=cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=list(language_codes),
                model=self.model,
                features=features,
            ),
            files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                inline_response_config=cloud_speech.InlineOutputConfig()
            ),
        )

    def submit(self, gcs_uri: str, **kwargs: Any) -> str:
        """Submit recognition and return its operation name."""
        request = self.build_request(gcs_uri, **kwargs)
        operation = self.client.batch_recognize(request=request)
        job_id = operation.operation.name
        self._job_parameters[job_id] = {
            "location": self.location,
            "model": self.model,
            "language_codes": list(request.config.language_codes),
            "word_time_offsets": request.config.features.enable_word_time_offsets,
            "diarization": request.config.features._pb.HasField("diarization_config"),
        }
        return job_id

    def get_status(self, job_id: str, details: bool = True) -> AsyncJobStatus:
        return _status_from_operation(self._get_operation(job_id), details=details)

    def list_jobs(self) -> list[AsyncJobStatus]:
        operations = self.operations_client.list_operations(self.operations_parent, "")
        return [
            _status_from_operation(operation, details=True)
            for operation in operations
            if _is_batch_recognize_operation(operation)
        ]

    def get_result(self, job_id: str) -> ToolOutput | None:
        operation = self._get_operation(job_id)
        if not _status_from_operation(operation, details=False).is_done:
            return None
        try:
            if operation.HasField("error"):
                raise GcpSpeechToTextError(job_id, operation.error.message)
            if not operation.HasField("response"):
                raise GcpSpeechToTextError(job_id, "completed without a response")
            native = cloud_speech.BatchRecognizeResponse.deserialize(operation.response.value)
            return self._to_tool_output(job_id, operation, native)
        finally:
            self._delete_operation(job_id)
            self._job_parameters.pop(job_id, None)

    def cleanup(self, job_id: str) -> None:
        try:
            operation = self._get_operation(job_id)
        except KeyError:
            return
        if not operation.done:
            self.operations_client.cancel_operation(job_id)
            started = time.monotonic()
            while not operation.done:
                if self.timeout is not None and time.monotonic() - started > self.timeout:
                    raise GcpSpeechToTextError(job_id, f"cleanup did not finish within {self.timeout} seconds")
                time.sleep(self.polling_interval)
                operation = self._get_operation(job_id)
        self._delete_operation(job_id)
        self._job_parameters.pop(job_id, None)

    def process(self, gcs_uri: str, **kwargs: Any) -> ToolOutput:
        """Submit, wait, convert to Transcript, and clean up."""
        job_id = self.submit(gcs_uri, **kwargs)
        started = time.monotonic()
        while not self.is_done(job_id):
            logging.info("GCP Speech-to-Text operation %s is still running", job_id)
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                self.cleanup(job_id)
                raise GcpSpeechToTextError(job_id, f"did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
        result = self.get_result(job_id)
        if result is None:
            raise GcpSpeechToTextError(job_id, "finished without a transcript")
        return result

    @staticmethod
    def native_to_tool_output(native: Any) -> ToolOutput:
        if not isinstance(native, cloud_speech.BatchRecognizeResponse):
            raise TypeError("native must be a BatchRecognizeResponse")
        return ToolOutput(tool_name="gcp_speech_to_text", tool_version=_version(), output=gcp_speech_to_transcript(native))

    def _to_tool_output(
        self,
        job_id: str,
        operation: operations_pb2.Operation,
        native: cloud_speech.BatchRecognizeResponse,
    ) -> ToolOutput:
        metadata = _metadata(operation)
        tool_private = None
        if self.include_tool_private:
            tool_private = {"operation_name": job_id, "native_response": _message_to_dict(native)}
        return ToolOutput(
            tool_name="gcp_speech_to_text",
            tool_version=_version(),
            parameters=self._job_parameters.pop(job_id, {"location": self.location, "model": self.model}),
            queue_time=_timestamp(metadata.create_time),
            start_time=_timestamp(metadata.create_time),
            end_time=_timestamp(metadata.update_time) or time.time(),
            output=gcp_speech_to_transcript(native),
            tool_private=tool_private,
        )

    def _get_operation(self, job_id: str) -> operations_pb2.Operation:
        try:
            return self.operations_client.get_operation(job_id)
        except NotFound as exc:
            raise KeyError(job_id) from exc

    def _delete_operation(self, job_id: str) -> None:
        try:
            self.operations_client.delete_operation(job_id)
        except (KeyError, NotFound):
            pass


def _status_from_operation(operation: operations_pb2.Operation, *, details: bool) -> AsyncJobStatus:
    metadata = _metadata(operation)
    status = (
        AsyncStatusCode.FAILED if operation.done and operation.HasField("error")
        else AsyncStatusCode.SUCCEEDED if operation.done
        else AsyncStatusCode.IN_PROGRESS if metadata.progress_percent
        else AsyncStatusCode.QUEUED
    )
    values = {
        "job_id": operation.name,
        "status": status,
        "progress": metadata.progress_percent,
        "message": operation.error.message if operation.HasField("error") else None,
    }
    if not details:
        return AsyncJobStatus(**values)
    return GcpSpeechJobStatus(
        **values,
        operation_name=operation.name,
        resource=metadata.resource or None,
        input_gcs_uris=[item.uri for item in metadata.batch_recognize_request.files],
    )


def _metadata(operation: operations_pb2.Operation) -> cloud_speech.OperationMetadata:
    metadata = cloud_speech.OperationMetadata()
    if operation.HasField("metadata"):
        operation.metadata.Unpack(cloud_speech.OperationMetadata.pb(metadata))
    return metadata


def _is_batch_recognize_operation(operation: operations_pb2.Operation) -> bool:
    metadata = _metadata(operation)
    return (
        metadata.method == "google.cloud.speech.v2.Speech.BatchRecognize"
        or operation.HasField("response")
        and operation.response.type_url.endswith("google.cloud.speech.v2.BatchRecognizeResponse")
    )


def _message_to_dict(message: Any) -> dict[str, Any]:
    return MessageToDict(type(message).pb(message), preserving_proto_field_name=True)


def _timestamp(value: Any) -> float | None:
    return value.timestamp() if value else None


def _version() -> str:
    from . import __version__
    return __version__
