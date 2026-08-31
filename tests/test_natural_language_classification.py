"""Tests for native Google Natural Language classification requests."""

import unittest

from google.cloud import language_v1

from ampav.gcp import GcpNaturalLanguageClassification


class _FakeLanguageClient:
    def __init__(self) -> None:
        self.request = None
        self.timeout = None
        self.response = language_v1.ClassifyTextResponse()

    def classify_text(self, *, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return self.response


class GcpNaturalLanguageClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeLanguageClient()
        self.tool = GcpNaturalLanguageClassification(client=self.client)

    def test_default_request_uses_v2_model_and_v2_taxonomy(self) -> None:
        request = self.tool.build_request("A transcript with enough words to classify.")

        options = request.classification_model_options
        self.assertTrue(options._pb.HasField("v2_model"))
        self.assertFalse(options._pb.HasField("v1_model"))
        self.assertEqual(
            options.v2_model.content_categories_version,
            language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2,
        )

    def test_v1_request_preserves_native_model_choice(self) -> None:
        request = self.tool.build_request(
            "A transcript with enough words to classify.", model="v1"
        )

        options = request.classification_model_options
        self.assertTrue(options._pb.HasField("v1_model"))
        self.assertFalse(options._pb.HasField("v2_model"))

    def test_v2_model_can_return_v1_taxonomy(self) -> None:
        request = self.tool.build_request(
            "A transcript with enough words to classify.",
            content_categories_version="v1",
        )

        self.assertEqual(
            request.classification_model_options.v2_model.content_categories_version,
            language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V1,
        )

    def test_process_returns_native_response_and_forwards_timeout(self) -> None:
        response = self.tool.process(
            "A transcript with enough words to classify.", timeout=12
        )

        self.assertIs(response, self.client.response)
        self.assertEqual(self.client.timeout, 12)
        self.assertIsInstance(self.client.request, language_v1.ClassifyTextRequest)

    def test_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "model must"):
            self.tool.build_request("Some text", model="v3")


if __name__ == "__main__":
    unittest.main()
