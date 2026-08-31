"""Tests for retained Speech-to-Text experiment metadata."""

from types import SimpleNamespace
import unittest

from experiments.speech_to_text import (
    _file_errors,
    _operation_details,
    _response_observations,
)


class SpeechToTextExperimentTest(unittest.TestCase):
    def test_operation_details_uses_native_operation_name(self) -> None:
        operation = SimpleNamespace(
            operation=SimpleNamespace(name="operations/test-operation"),
            metadata=None,
        )

        self.assertEqual(
            _operation_details(operation),
            {"name": "operations/test-operation"},
        )

    def test_observations_report_native_file_errors(self) -> None:
        response = {
            "results": {
                "gs://bucket/audio.m4a": {
                    "error": {
                        "code": 13,
                        "message": "An internal error occurred.",
                    }
                }
            },
            "total_billed_duration": "0s",
        }
        observations = _response_observations(response)

        self.assertEqual(
            _file_errors(response),
            [
                (
                    "gs://bucket/audio.m4a",
                    {"code": 13, "message": "An internal error occurred."},
                )
            ],
        )
        self.assertIn("File errors: 1", observations)
        self.assertIn("code 13 - An internal error occurred.", observations)


if __name__ == "__main__":
    unittest.main()
