"""Tests for the integrated Speech-to-Text async tool."""

from datetime import timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

from google.cloud.speech_v2.types import cloud_speech
from google.longrunning import operations_pb2
from google.protobuf.any_pb2 import Any as AnyMessage

from ampav.core.async_tool import AsyncStatusCode
from ampav.core.schema import ToolOutput, Transcript
from ampav.gcp import GcpSpeechJobStatus, GcpSpeechToTextBatch
from ampav.gcp.errors import GcpSpeechToTextError
from ampav_gcp_cli.speech_to_text import build_cli_parser, main as cli_main
from ampav_gcp_pipeline import transcribe_file


JOB_ID = "projects/test-project/locations/us/operations/test-job"
OPEN_DOOR = Path(__file__).parents[1] / "examples" / "data" / "OpenDoor.wav"


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


class _FakeBlob:
    def __init__(self, client, bucket: str, object_name: str) -> None:
        self.client = client
        self.bucket = bucket
        self.object_name = object_name

    def upload_from_filename(self, source: str) -> None:
        self.client.uploads.append((source, self.bucket, self.object_name))

    def delete(self) -> None:
        self.client.deleted.append((self.bucket, self.object_name))


class _FakeBucket:
    def __init__(self, client, name: str) -> None:
        self.client = client
        self.name = name

    def blob(self, object_name: str) -> _FakeBlob:
        return _FakeBlob(self.client, self.name, object_name)


class _FakeStorageClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self, name)


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

    def test_transcribe_file_uploads_fixture_and_deletes_temporary_object(self) -> None:
        storage = _FakeStorageClient()
        speech = _FakeSpeechClient()

        output = transcribe_file(
            OPEN_DOOR,
            project_id="test-project",
            input_bucket="gs://bucket",
            input_object_name="speech-input/OpenDoor.wav",
            speech_client=speech,
            storage_client=storage,
            polling_interval=0.001,
        )

        self.assertEqual(output.output.text, "Please open the door.")
        self.assertEqual(storage.uploads, [(str(OPEN_DOOR.resolve()), "bucket", "speech-input/OpenDoor.wav")])
        self.assertEqual(storage.deleted, [("bucket", "speech-input/OpenDoor.wav")])
        self.assertEqual(speech.request.files[0].uri, "gs://bucket/speech-input/OpenDoor.wav")

    def test_cli_parser_and_gcs_dispatch(self) -> None:
        args = build_cli_parser().parse_args(
            [
                "gs://bucket/OpenDoor.wav",
                "--project-id",
                "test-project",
                "--location",
                "us-central1",
                "--language-code",
                "en-US",
                "--no-diarization",
            ]
        )
        self.assertEqual(args.media, "gs://bucket/OpenDoor.wav")
        self.assertEqual(args.project_id, "test-project")
        self.assertEqual(args.language_code, ["en-US"])
        self.assertTrue(args.no_diarization)

        output = ToolOutput(
            tool_name="gcp_speech_to_text",
            tool_version="test",
            output=Transcript(text="Please open the door."),
        )
        with (
            patch("ampav_gcp_cli.speech_to_text.GcpSpeechToTextBatch") as tool_class,
            patch("builtins.print") as print_output,
        ):
            tool_class.return_value.process.return_value = output
            exit_code = cli_main(
                ["gs://bucket/OpenDoor.wav", "--project-id", "test-project", "--no-diarization"]
            )

        self.assertEqual(exit_code, 0)
        print_output.assert_called_once()
        tool_class.return_value.process.assert_called_once_with(
            "gs://bucket/OpenDoor.wav",
            language_codes=("en-US",),
            enable_word_time_offsets=True,
            enable_diarization=False,
        )


if __name__ == "__main__":
    unittest.main()
