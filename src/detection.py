"""AprilTag detection and tracker updates. Contains no camera or robot code."""

import cv2
import numpy as np
from pupil_apriltags import Detector

# pupil_apriltags 1.0.4 can crash the whole process (segmentation fault) when a
# detector that used more than one thread is destroyed, for example when the GUI
# closes. In testing, 2, 4 and 12 threads all crashed and 1 thread never did. One
# thread takes about 74 ms per 1080p frame instead of about 43 ms with 12.
DETECTOR_THREADS = 1


class TagPipeline:
    """Detect tags in a frame and update trackers.

    Tracker poses are expressed in the camera frame, or in the reference tag's
    frame when ``reference_id`` is set.
    """

    def __init__(
        self,
        trackers,
        camera_params,
        tag_size_m,
        tag_family="tag36h11",
        decision_margin=3.0,
        reference_id=None,
        tag_sizes_m=None,
    ):
        """
        Args:
            trackers:        Tracker instances to update each frame.
            camera_params:   (fx, fy, cx, cy) in pixels.
            tag_size_m:      Edge length of the tag's black square, in metres.
            tag_sizes_m:     Optional dict of tag_id -> edge length in metres for
                             tags whose size differs from tag_size_m. These tags
                             are also accepted even if no tracker uses them.
            tag_family:      AprilTag family name.
            decision_margin: Minimum detector confidence (higher = stricter).
            reference_id:    Fixed tag defining the output frame, or None for
                             the camera frame.
        """
        self.trackers = trackers
        self.camera_params = camera_params
        self.tag_size_m = tag_size_m
        self.tag_sizes_m = dict(tag_sizes_m or {})
        self.decision_margin = decision_margin
        self.reference_id = reference_id
        self.last_reference_pose = None  # persists across brief occlusions

        self.allowed_ids = {tag_id for tracker in trackers for tag_id in tracker.ids}
        if reference_id is not None:
            self.allowed_ids.add(reference_id)
        self.allowed_ids.update(self.tag_sizes_m)

        self.detector = Detector(
            families=tag_family,
            quad_decimate=1.0,
            nthreads=DETECTOR_THREADS,
            refine_edges=1,
            quad_sigma=0.2,
            decode_sharpening=1.0,
        )

    @property
    def use_reference_frame(self):
        """Return True when poses are expressed in the reference tag's frame."""
        return self.reference_id is not None

    def process_frame(self, frame):
        """Detect tags, update the reference pose, refresh tracker poses.

        Returns:
            dict of tag_id -> raw Detection (camera frame).
        """
        tag_dict = self.detect_tags(frame)

        if self.use_reference_frame and self.reference_id in tag_dict:
            self.last_reference_pose = tag_dict[self.reference_id]

        tags_out = self._transform_to_output_frame(tag_dict)
        for tracker in self.trackers:
            tracker.updatePose(tags_out)

        return tag_dict

    def detect_tags(self, frame):
        """Run AprilTag detection and filter by ID and decision margin.

        The detector takes a single tag size, so every tag is solved at tag_size_m
        and tags of another size have their position rescaled. Position scales
        exactly with tag size and rotation doesn't change.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size_m,
        )
        tags = {
            tag.tag_id: tag
            for tag in raw_tags
            if tag.decision_margin > self.decision_margin
            and tag.tag_id in self.allowed_ids
        }
        for tag_id, size in self.tag_sizes_m.items():
            if tag_id in tags:
                tags[tag_id].pose_t = tags[tag_id].pose_t * (size / self.tag_size_m)
        return tags

    def _transform_to_output_frame(self, tag_dict):
        """Return tag poses in either the camera or the reference-tag frame."""
        if not self.use_reference_frame:
            return {
                tag_id: {
                    "pos": np.asarray(tag.pose_t, dtype=np.float32).reshape(3),
                    "rot": np.asarray(tag.pose_R, dtype=np.float32),
                }
                for tag_id, tag in tag_dict.items()
            }

        if self.last_reference_pose is None:
            return {}

        R_ref_inv = self.last_reference_pose.pose_R.T
        t_ref = self.last_reference_pose.pose_t
        return {
            tag_id: {
                "pos": (R_ref_inv @ (tag.pose_t - t_ref)).flatten().astype(np.float32),
                "rot": (R_ref_inv @ tag.pose_R).astype(np.float32),
            }
            for tag_id, tag in tag_dict.items()
        }
