"""Tests for native synchronous Gemini generation."""

import unittest

from google.genai import types

from ampav.gcp import GcpGeminiGenerateContent


class _FakeModels:
    def __init__(self) -> None:
        self.kwargs = None
        self.response = types.GenerateContentResponse(response_id="test-response")

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class GcpGeminiGenerateContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.tool = GcpGeminiGenerateContent(
            "test-project", client=self.client, model="test-model"
        )

    def test_build_config_enables_structured_json(self) -> None:
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        config = self.tool.build_config(
            system_instruction="Stay grounded.",
            response_json_schema=schema,
            temperature=0.1,
            max_output_tokens=512,
        )

        self.assertEqual(config.system_instruction, "Stay grounded.")
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, schema)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_output_tokens, 512)
        self.assertTrue(config.automatic_function_calling.disable)

    def test_process_returns_native_response_and_model_choice(self) -> None:
        response = self.tool.process("Transcript text")

        self.assertIs(response, self.client.models.response)
        self.assertEqual(self.client.models.kwargs["model"], "test-model")
        self.assertEqual(self.client.models.kwargs["contents"], "Transcript text")
        self.assertIsInstance(
            self.client.models.kwargs["config"], types.GenerateContentConfig
        )

    def test_unstructured_generation_does_not_force_mime_type(self) -> None:
        config = self.tool.build_config(response_json_schema=None)

        self.assertIsNone(config.response_mime_type)
        self.assertIsNone(config.response_json_schema)

    def test_rejects_empty_contents(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.tool.process("  ")


if __name__ == "__main__":
    unittest.main()
