"""RealSense color capture and optional MP4 recording."""

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from src.config import CameraConfig


class CameraResetError(RuntimeError):
    """The camera reset itself, so its settings are back to the factory defaults."""


class RealSenseCamera:
    """Start the color stream and expose its intrinsics."""

    def __init__(self, config: CameraConfig):
        self.pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(
            rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps
        )
        profile = self.pipeline.start(stream_config)

        self.color_sensor = profile.get_device().query_sensors()[1]
        self.auto_exposure_on = None  # Unknown until this object sets it.
        self.last_frame_number = None
        if config.exposure is None:
            self.set_auto_exposure(True)
        else:
            self.set_exposure(config.exposure)

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

    def exposure_range(self):
        """Return the lowest and highest manual exposure the camera allows."""
        option_range = self.color_sensor.get_option_range(rs.option.exposure)
        return option_range.min, option_range.max

    def current_exposure(self):
        """Return the camera's current exposure setting."""
        return self.color_sensor.get_option(rs.option.exposure)

    def set_exposure(self, value):
        """Use a manual exposure, turning auto exposure off first if needed.

        Each setting change is a command the camera must answer while streaming,
        and a burst of them can make it stop responding, so none are sent twice.
        """
        if self.auto_exposure_on is not False:
            self.set_auto_exposure(False)
        self.color_sensor.set_option(rs.option.exposure, value)

    def set_auto_exposure(self, enabled):
        """Turn auto exposure on or off."""
        self.color_sensor.set_option(rs.option.enable_auto_exposure, 1 if enabled else 0)
        self.auto_exposure_on = bool(enabled)

    def read(self):
        """Block until a color frame arrives; return it as a BGR array, or None.

        Raises:
            CameraResetError: The camera reset itself. Intel's library restarts the
                stream on its own afterwards, but the camera is back on its default
                settings and the settings handle here no longer works, so the camera
                must be reopened. A reset shows up as the frame counter starting
                again from zero.
        """
        color_frame = self.pipeline.wait_for_frames().get_color_frame()
        if not color_frame:
            return None
        frame_number = color_frame.get_frame_number()
        if self.last_frame_number is not None and frame_number < self.last_frame_number:
            raise CameraResetError("the camera reset itself and lost its settings")
        self.last_frame_number = frame_number
        return np.asanyarray(color_frame.get_data())

    def close(self):
        """Stop the color stream."""
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
