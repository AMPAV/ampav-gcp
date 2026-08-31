"""Native asynchronous Google Cloud Video Intelligence annotation."""

from collections.abc import Sequence
from typing import Any

from google.cloud import videointelligence_v1 as videointelligence


DEFAULT_FEATURES = (
    videointelligence.Feature.SHOT_CHANGE_DETECTION,
    videointelligence.Feature.LABEL_DETECTION,
    videointelligence.Feature.TEXT_DETECTION,
    videointelligence.Feature.FACE_DETECTION,
    videointelligence.Feature.PERSON_DETECTION,
    videointelligence.Feature.OBJECT_TRACKING,
)


class GcpVideoIntelligence:
    """Annotate one GCS video and return Google's native operation/response.

    Uploads, persistence, cleanup, feature-specific interpretation, and AMPAV
    schema conversion remain client responsibilities.
    """

    def __init__(
        self,
        *,
        location: str = "us-east1",
        model: str = "builtin/stable",
        client: videointelligence.VideoIntelligenceServiceClient | None = None,
    ) -> None:
        if not location:
            raise ValueError("location must not be empty")
        if model not in {"builtin/stable", "builtin/latest"}:
            raise ValueError("model must be 'builtin/stable' or 'builtin/latest'")
        self.location = location
        self.model = model
        self.client = client or videointelligence.VideoIntelligenceServiceClient()

    def build_request(
        self,
        gcs_uri: str,
        *,
        features: Sequence[videointelligence.Feature] = DEFAULT_FEATURES,
        include_bounding_boxes: bool = True,
        include_attributes: bool = True,
        include_pose_landmarks: bool = False,
    ) -> videointelligence.AnnotateVideoRequest:
        """Build a native multi-feature annotation request."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError("gcs_uri must start with gs://")
        if not features:
            raise ValueError("features must contain at least one value")
        if videointelligence.Feature.FEATURE_UNSPECIFIED in features:
            raise ValueError("features must not contain FEATURE_UNSPECIFIED")

        requested = set(features)
        context = videointelligence.VideoContext()
        if videointelligence.Feature.LABEL_DETECTION in requested:
            context.label_detection_config = videointelligence.LabelDetectionConfig(
                label_detection_mode=videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE,
                model=self.model,
            )
        if videointelligence.Feature.SHOT_CHANGE_DETECTION in requested:
            context.shot_change_detection_config = (
                videointelligence.ShotChangeDetectionConfig(model=self.model)
            )
        if videointelligence.Feature.TEXT_DETECTION in requested:
            context.text_detection_config = videointelligence.TextDetectionConfig(
                model=self.model
            )
        if videointelligence.Feature.FACE_DETECTION in requested:
            context.face_detection_config = videointelligence.FaceDetectionConfig(
                model=self.model,
                include_bounding_boxes=include_bounding_boxes,
                include_attributes=include_attributes,
            )
        if videointelligence.Feature.PERSON_DETECTION in requested:
            context.person_detection_config = videointelligence.PersonDetectionConfig(
                include_bounding_boxes=include_bounding_boxes,
                include_attributes=include_attributes,
                include_pose_landmarks=include_pose_landmarks,
            )
        if videointelligence.Feature.OBJECT_TRACKING in requested:
            context.object_tracking_config = videointelligence.ObjectTrackingConfig(
                model=self.model
            )

        return videointelligence.AnnotateVideoRequest(
            input_uri=gcs_uri,
            features=list(features),
            video_context=context,
            location_id=self.location,
        )

    def submit(
        self,
        gcs_uri: str,
        *,
        features: Sequence[videointelligence.Feature] = DEFAULT_FEATURES,
        include_bounding_boxes: bool = True,
        include_attributes: bool = True,
        include_pose_landmarks: bool = False,
    ) -> Any:
        """Submit a request and return Google's native long-running operation."""
        request = self.build_request(
            gcs_uri,
            features=features,
            include_bounding_boxes=include_bounding_boxes,
            include_attributes=include_attributes,
            include_pose_landmarks=include_pose_landmarks,
        )
        return self.client.annotate_video(request=request)

    def process(
        self,
        gcs_uri: str,
        *,
        features: Sequence[videointelligence.Feature] = DEFAULT_FEATURES,
        include_bounding_boxes: bool = True,
        include_attributes: bool = True,
        include_pose_landmarks: bool = False,
        timeout: float | None = None,
    ) -> videointelligence.AnnotateVideoResponse:
        """Submit and wait for a native annotation response."""
        operation = self.submit(
            gcs_uri,
            features=features,
            include_bounding_boxes=include_bounding_boxes,
            include_attributes=include_attributes,
            include_pose_landmarks=include_pose_landmarks,
        )
        return operation.result(timeout=timeout)
