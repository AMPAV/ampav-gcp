"""Tests for the integrated Speech-to-Text async tool."""

from datetime import timedelta
import unittest

from google.cloud.speech_v2.types import cloud_speech
from google.longrunning import operations_pb2
from google.protobuf.any_pb2 import Any as AnyMessage

from ampav.core.async_tool import AsyncStatusCode
from ampav.core.schema import ToolOutput, Transcript
from ampav.gcp import GcpSpeechJobStatus, GcpSpeechToTextBatch
from ampav.gcp.errors import GcpSpeechToTextError


JOB_ID = "projects/test-project/locations/us/operations/test-job"


def native_response() -> cloud_speech.BatchRecognizeResponse:
    words = [
        cloud_speech.WordInfo(
            word=word,
            start_offset=timedelta(seconds=start),
            end_offset=timedelta(seconds=end),
            speaker_label="1",
        )
        for word, start, end in [
            ("Please", 0.0, 0.4),
            ("open", 0.45, 0.8),
            ("the", 0.85, 1.0),
            ("door.", 1.05, 1.649),
        ]
    ]
    result = cloud_speech.SpeechRecognitionResult(
        alternatives=[cloud_speech.SpeechRecognitionAlternative(transcript="Please open the door.", words=words)],
        language_code="en-US",
    )
    return cloud_speech.BatchRecognizeResponse(
        results={
            "gs://bucket/OpenDoor.wav": cloud_speech.BatchRecognizeFileResult(
                uri="gs://bucket/OpenDoor.wav",
                inline_result=cloud_speech.InlineResult(
                    transcript=cloud_speech.BatchRecognizeResults(results=[result])
                ),
            )
        }
    )


def raw_operation(*, done: bool, progress: int = 0, failed: bool = False) -> operations_pb2.Operation:
    metadata = cloud_speech.OperationMetadata(
        resource="projects/test-project/locations/us/recognizers/_",
        method="google.cloud.speech.v2.Speech.BatchRecognize",
        progress_percent=progress,
        batch_recognize_request=cloud_speech.BatchRecognizeRequest(
            files=[cloud_speech.BatchRecognizeFileMetadata(uri="gs://bucket/OpenDoor.wav")]
        ),
    )
    metadata_any = AnyMessage()
    metadata_any.Pack(cloud_speech.OperationMetadata.pb(metadata))
    operation = operations_pb2.Operation(name=JOB_ID, done=done, metadata=metadata_any)
    if failed:
        operation.error.code = 13
        operation.error.message = "test failure"
    elif done:
        response_any = AnyMessage()
        response_any.Pack(cloud_speech.BatchRecognizeResponse.pb(native_response()))
        operation.response.CopyFrom(response_any)
    return operation


class _FakeOperationsClient:
    def __init__(self) -> None:
        self.operation = raw_operation(done=True, progress=100)
        self.deleted: list[str] = []
        self.cancelled: list[str] = []

    def get_operation(self, name):
        if name in self.deleted:
            raise KeyError(name)
        return self.operation

    def list_operations(self, name, filter_):
        return [self.operation]

    def delete_operation(self, name):
        self.deleted.append(name)

    def cancel_operation(self, name):
        self.cancelled.append(name)
        self.operation.done = True


class _FakeSpeechClient:
    def __init__(self) -> None:
        self.request = None
        self.transport = type("Transport", (), {"operations_client": _FakeOperationsClient()})()

    def batch_recognize(self, *, request):
        self.request = request
        return type("Future", (), {"operation": self.transport.operations_client.operation})()


class GcpSpeechToTextBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeSpeechClient()
        self.tool = GcpSpeechToTextBatch("test-project", client=self.client, polling_interval=0.001)

    def test_submit_returns_operation_name_and_native_request(self) -> None:
        job_id = self.tool.submit("gs://bucket/OpenDoor.wav")

        self.assertEqual(job_id, JOB_ID)
        self.assertEqual(self.client.request.config.model, "chirp_3")
        self.assertTrue(self.client.request.config.features.enable_word_time_offsets)
        self.assertTrue(self.client.request.config.features._pb.HasField("diarization_config"))

    def test_status_preserves_progress_and_input_details(self) -> None:
        self.client.transport.operations_client.operation = raw_operation(done=False, progress=42)

        status = self.tool.get_status(JOB_ID)

        self.assertIsInstance(status, GcpSpeechJobStatus)
        self.assertEqual(status.status, AsyncStatusCode.IN_PROGRESS)
        self.assertEqual(status.progress, 42)
        self.assertEqual(status.input_gcs_uris, ["gs://bucket/OpenDoor.wav"])

    def test_get_result_returns_tool_output_transcript_and_cleans_up(self) -> None:
        self.tool._job_parameters[JOB_ID] = {"model": "chirp_3"}

        output = self.tool.get_result(JOB_ID)

        self.assertIsInstance(output, ToolOutput)
        self.assertIsInstance(output.output, Transcript)
        self.assertEqual(output.output.text, "Please open the door.")
        self.assertEqual([word.to_str() for word in output.output.words], ["Please", "open", "the", "door."])
        self.assertEqual(output.output.media_duration, 1.649)
        self.assertEqual(output.output.words[0].speaker, "1")
        self.assertIsNone(output.output.words[0].confidence)
        self.assertEqual(self.client.transport.operations_client.deleted, [JOB_ID])

    def test_process_returns_normalized_output(self) -> None:
        output = self.tool.process("gs://bucket/OpenDoor.wav")

        self.assertEqual(output.tool_name, "gcp_speech_to_text")
        self.assertEqual(output.output.languages, ["en-US"])

    def test_failed_operation_raises_typed_error_and_cleans_up(self) -> None:
        self.client.transport.operations_client.operation = raw_operation(done=True, failed=True)

        with self.assertRaisesRegex(GcpSpeechToTextError, "test failure"):
            self.tool.get_result(JOB_ID)

        self.assertEqual(self.client.transport.operations_client.deleted, [JOB_ID])

    def test_cleanup_cancels_running_operation_then_deletes_record(self) -> None:
        self.client.transport.operations_client.operation = raw_operation(done=False, progress=42)

        self.tool.cleanup(JOB_ID)

        operations = self.client.transport.operations_client
        self.assertEqual(operations.cancelled, [JOB_ID])
        self.assertEqual(operations.deleted, [JOB_ID])


if __name__ == "__main__":
    unittest.main()
