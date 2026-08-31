"""Tests for native Speech-to-Text V2 batch requests."""

import unittest

from google.cloud.speech_v2.types import cloud_speech

from ampav.gcp import GcpSpeechToTextBatch


class _FakeOperation:
    def __init__(self, response: cloud_speech.BatchRecognizeResponse) -> None:
        self.response = response
        self.timeout = None

    def result(self, timeout: float | None = None) -> cloud_speech.BatchRecognizeResponse:
        self.timeout = timeout
        return self.response


class _FakeSpeechClient:
    def __init__(self) -> None:
        self.request = None
        self.operation = _FakeOperation(cloud_speech.BatchRecognizeResponse())

    def batch_recognize(self, *, request: cloud_speech.BatchRecognizeRequest):
        self.request = request
        return self.operation


class GcpSpeechToTextBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeSpeechClient()
        self.tool = GcpSpeechToTextBatch(
            "test-project",
            client=self.client,
        )

    def test_build_request_preserves_native_chirp_configuration(self) -> None:
        request = self.tool.build_request("gs://bucket/audio.m4a")

        self.assertEqual(
            request.recognizer,
            "projects/test-project/locations/us/recognizers/_",
        )
        self.assertEqual(request.config.model, "chirp_3")
        self.assertEqual(list(request.config.language_codes), ["en-US"])
        self.assertTrue(request.config.features.enable_word_time_offsets)
        self.assertTrue(request.config.features._pb.HasField("diarization_config"))
        self.assertEqual(request.files[0].uri, "gs://bucket/audio.m4a")
        self.assertTrue(
            request.recognition_output_config._pb.HasField(
                "inline_response_config"
            )
        )

    def test_optional_features_can_be_disabled(self) -> None:
        request = self.tool.build_request(
            "gs://bucket/audio.m4a",
            enable_word_time_offsets=False,
            enable_diarization=False,
        )

        self.assertFalse(request.config.features.enable_word_time_offsets)
        self.assertFalse(request.config.features._pb.HasField("diarization_config"))

    def test_process_returns_native_response_and_forwards_timeout(self) -> None:
        response = self.tool.process("gs://bucket/audio.m4a", timeout=42)

        self.assertIs(response, self.client.operation.response)
        self.assertEqual(self.client.operation.timeout, 42)
        self.assertIsInstance(
            self.client.request,
            cloud_speech.BatchRecognizeRequest,
        )

    def test_rejects_non_gcs_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start with gs://"):
            self.tool.build_request("audio.m4a")


if __name__ == "__main__":
    unittest.main()
