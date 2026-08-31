"""Tests for Gemini aboutness experiment schemas."""

import unittest

from experiments.gemini_aboutness import MODES, _response_schema


class GeminiAboutnessExperimentTest(unittest.TestCase):
    def test_combined_schema_requires_all_candidate_outputs(self) -> None:
        schema = _response_schema("combined")

        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            set(schema["properties"]),
            {
                "summary",
                "subjects",
                "topics",
                "themes",
                "categories",
                "keyphrases",
                "named_entities",
                "controlled_labels",
            },
        )

    def test_focused_schemas_require_only_selected_term(self) -> None:
        for mode in MODES[1:]:
            with self.subTest(mode=mode):
                schema = _response_schema(mode)
                self.assertEqual(schema["required"], [mode])
                self.assertEqual(list(schema["properties"]), [mode])


if __name__ == "__main__":
    unittest.main()
