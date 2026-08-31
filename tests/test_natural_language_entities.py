"""Tests for native Google Natural Language entity requests."""

import unittest

from google.cloud import language_v1

from ampav.gcp import GcpNaturalLanguageEntities


class _FakeLanguageClient:
    def __init__(self) -> None:
        self.request = None
        self.timeout = None
        self.response = language_v1.AnalyzeEntitiesResponse(language="en")

    def analyze_entities(self, *, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return self.response


class GcpNaturalLanguageEntitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeLanguageClient()
        self.tool = GcpNaturalLanguageEntities(client=self.client)

    def test_build_request_preserves_native_text_configuration(self) -> None:
        request = self.tool.build_request("Indiana University", language="en")

        self.assertEqual(request.document.content, "Indiana University")
        self.assertEqual(request.document.type_, language_v1.Document.Type.PLAIN_TEXT)
        self.assertEqual(request.document.language, "en")
        self.assertEqual(request.encoding_type, language_v1.EncodingType.UTF8)

    def test_language_can_be_auto_detected(self) -> None:
        request = self.tool.build_request("Indiana University")

        self.assertEqual(request.document.language, "")

    def test_process_returns_native_response_and_forwards_timeout(self) -> None:
        response = self.tool.process("Indiana University", timeout=12)

        self.assertIs(response, self.client.response)
        self.assertEqual(self.client.timeout, 12)
        self.assertIsInstance(self.client.request, language_v1.AnalyzeEntitiesRequest)

    def test_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.tool.build_request("  ")


if __name__ == "__main__":
    unittest.main()
