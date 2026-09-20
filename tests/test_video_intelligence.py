"""Tests for native combined Video Intelligence requests."""

import unittest

from google.cloud import videointelligence_v1 as videointelligence

from ampav.gcp import GcpVideoIntelligence
from ampav.gcp.video_intelligence import DEFAULT_FEATURES


class _FakeOperation:
    def __init__(self) -> None:
        self.response = videointelligence.AnnotateVideoResponse()
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return self.response


class _FakeVideoClient:
    def __init__(self) -> None:
        self.request = None
        self.operation = _FakeOperation()

    def annotate_video(self, *, request):
        self.request = request
        return self.operation


class GcpVideoIntelligenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeVideoClient()
        self.tool = GcpVideoIntelligence(client=self.client)

    def test_default_request_combines_selected_features(self) -> None:
        request = self.tool.build_request("gs://bucket/video.mp4")

        self.assertEqual(list(request.features), list(DEFAULT_FEATURES))
        self.assertEqual(request.location_id, "us-east1")
        self.assertEqual(request.input_uri, "gs://bucket/video.mp4")
        context = request.video_context
        self.assertEqual(context.label_detection_config.model, "builtin/stable")
        self.assertEqual(
            context.label_detection_config.label_detection_mode,
            videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE,
        )
        self.assertTrue(context.face_detection_config.include_bounding_boxes)
        self.assertTrue(context.face_detection_config.include_attributes)
        self.assertTrue(context.person_detection_config.include_bounding_boxes)
        self.assertTrue(context.person_detection_config.include_attributes)
        self.assertFalse(context.person_detection_config.include_pose_landmarks)

    def test_pose_landmarks_are_opt_in(self) -> None:
        request = self.tool.build_request(
            "gs://bucket/video.mp4", include_pose_landmarks=True
        )

        self.assertTrue(request.video_context.person_detection_config.include_pose_landmarks)

    def test_single_feature_only_sets_relevant_context(self) -> None:
        request = self.tool.build_request(
            "gs://bucket/video.mp4",
            features=[videointelligence.Feature.SHOT_CHANGE_DETECTION],
        )

        self.assertTrue(
            request.video_context._pb.HasField("shot_change_detection_config")
        )
        self.assertFalse(request.video_context._pb.HasField("face_detection_config"))

    def test_process_returns_native_response_and_forwards_timeout(self) -> None:
        response = self.tool.process("gs://bucket/video.mp4", timeout=42)

        self.assertIs(response, self.client.operation.response)
        self.assertEqual(self.client.operation.timeout, 42)
        self.assertIsInstance(
            self.client.request, videointelligence.AnnotateVideoRequest
        )

    def test_rejects_non_gcs_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start with gs://"):
            self.tool.build_request("video.mp4")


if __name__ == "__main__":
    unittest.main()
