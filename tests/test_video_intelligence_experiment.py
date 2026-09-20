"""Tests for retained Video Intelligence observation summaries."""

import unittest

from experiments.video_intelligence import _response_observations


class VideoIntelligenceExperimentTest(unittest.TestCase):
    def test_observations_summarize_feature_arrays(self) -> None:
        response = {
            "annotation_results": [
                {
                    "shot_annotations": [{"start_time_offset": "0s"}],
                    "text_annotations": [{"segments": [{"confidence": 0.9}]}],
                    "face_detection_annotations": [{"tracks": [{"confidence": 0.8}]}],
                    "person_detection_annotations": [{"tracks": [{"confidence": 0.7}]}],
                    "object_annotations": [{"confidence": 0.6}],
                },
                {"error": {"code": 2, "message": "Calculator failure."}},
            ]
        }

        observations = _response_observations(response)

        self.assertIn("Shot segments: 1", observations)
        self.assertIn("OCR text segments: 1", observations)
        self.assertIn("Face tracks: 1", observations)
        self.assertIn("Person tracks: 1", observations)
        self.assertIn("Object tracks: 1", observations)
        self.assertIn("Calculator failure.", observations)


if __name__ == "__main__":
    unittest.main()
