"""RealSense color capture and optional MP4 recording."""

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from src.config import CameraConfig


class RealSenseCamera:
    """Start the color stream and expose its intrinsics."""

    def __init__(self, config: CameraConfig):
        self.pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(
            rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps
        )
        profile = self.pipeline.start(stream_config)

        if config.exposure is not None:
            color_sensor = profile.get_device().query_sensors()[1]
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, config.exposure)

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.camera_params = (
            intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
        )
        self.K = np.array(
            [[intrinsics.fx, 0,             intrinsics.ppx],
             [0,             intrinsics.fy, intrinsics.ppy],
             [0,             0,             1             ]],
            dtype=np.float32,
        )
        # D435 distortion is negligible at typical working distances.
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def read(self):
        """Block until a color frame arrives; return it as a BGR array, or None."""
        color_frame = self.pipeline.wait_for_frames().get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        self.pipeline.stop()


class VideoRecorder:
    """Write frames to an MP4, opening the file on the first frame."""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError(f"Could not open video writer for {self.path}")
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
