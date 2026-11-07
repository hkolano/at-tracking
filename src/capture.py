"""Background camera loop and recording for the GUI.

The worker owns the camera and detector on its own thread. The GUI reads the
latest result with ``get_state()`` and messages with ``get_notices()``.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
from scipy.spatial.transform import Rotation as R

from src import annotate
from src.camera import RealSenseCamera
from src.data_collector import DataCollector
from src.detection import TagPipeline
from src.measurement import (
    RollingMedian,
    average_poses,
    build_output_paths,
    clean_id,
    relative_pose,
    write_average_csv,
)

# Warn when more than this share of a tag's frames were dropped as flips.
REJECTED_WARNING_FRACTION = 0.2
FPS_WINDOW_FRAMES = 30


@dataclass(frozen=True)
class CaptureState:
    """The latest processed frame, for display."""

    sequence: int                 # increases with every processed frame
    frame: object                 # annotated BGR image
    fps: float
    visible_ids: frozenset
    seen_ids: tuple               # every tag ID seen this session, sorted
    relative_positions: dict      # tag_id -> (3,) filtered position in the reference frame
    reference_id: int | None
    recording_remaining: float | None  # seconds left, or None when not recording
    exposure_range: tuple         # (lowest, highest) manual exposure
    exposure: float               # manual exposure, kept while auto is on
    auto_exposure: bool


@dataclass(frozen=True)
class Notice:
    """A message for the status bar."""

    text: str
    is_error: bool = False
    is_fatal: bool = False


@dataclass
class _Recording:
    """Everything collected during one recording."""

    bush_id: str
    cane_id: str
    seconds: float
    reference_id: int
    started_at: float
    deadline: float
    snapshot: object
    collector: DataCollector
    positions: dict = field(default_factory=dict)  # tag_id -> list of (3,) positions
    rotations: dict = field(default_factory=dict)  # tag_id -> list of (3, 3) rotations
    seen_ids: set = field(default_factory=set)


def tag_names_from_config(config):
    """Return tag_id -> display name, from the objects in a config."""
    names = {}
    for tracked_object in config.objects:
        has_many_tags = len(tracked_object.tag_ids) > 1
        for tag_id in tracked_object.tag_ids:
            names[tag_id] = (
                f"{tracked_object.name}_tag{tag_id}" if has_many_tags else tracked_object.name
            )
    return names


class CaptureWorker:
    """Run the camera loop on a background thread."""

    def __init__(self, config, output_dir="data", camera_factory=RealSenseCamera):
        """Prepare the worker. Call start() to open the camera."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.camera_factory = camera_factory
        self.names = tag_names_from_config(config)

        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._notices = queue.Queue()

        self._reference_id = config.reference_tag
        self._recording_request = None  # (seconds, bush_id, cane_id) waiting for the next frame
        self._recording = None
        self._state = None
        self._seen_ids = set()
        self._frame_times = deque(maxlen=FPS_WINDOW_FRAMES)

        self._position_filter = RollingMedian()
        self._filter_reference_id = None

        self._exposure = config.camera.exposure
        self._auto_exposure = config.camera.exposure is None
        self._exposure_range = (0.0, 0.0)
        self._exposure_request = None  # (auto, value), newest only

        self._camera = None
        self._pipeline = None

    def start(self):
        """Start the background thread."""
        self._thread.start()

    def stop(self, timeout=10.0):
        """Stop the thread and release the camera. A running recording is discarded."""
        self._stop_requested.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def set_reference(self, tag_id):
        """Set the reference tag. A running recording keeps its original reference."""
        with self._lock:
            self._reference_id = tag_id

    def set_exposure(self, value):
        """Ask for a manual exposure, unless a recording is running."""
        self._request_exposure(False, value)

    def set_auto_exposure(self, enabled):
        """Turn auto exposure on, or return to the last manual exposure."""
        self._request_exposure(enabled, None)

    def _request_exposure(self, auto, value):
        """Keep the newest exposure request for the capture thread to apply."""
        with self._lock:
            if self._recording_request is None and self._recording is None:
                self._exposure_request = (auto, value)

    def start_recording(self, seconds, bush_id, cane_id):
        """Ask for a recording to begin on the next frame."""
        if seconds <= 0:
            raise ValueError("Recording length must be positive")
        bush_id, cane_id = clean_id(bush_id), clean_id(cane_id)
        if not bush_id or not cane_id:
            raise ValueError("A bush ID and cane ID are required")
        with self._lock:
            if self._recording_request is None and self._recording is None:
                self._recording_request = (float(seconds), bush_id, cane_id)

    def is_recording(self):
        """Return True while a recording is requested or running."""
        with self._lock:
            return self._recording_request is not None or self._recording is not None

    def get_state(self):
        """Return the latest CaptureState, or None before the first frame."""
        with self._lock:
            return self._state

    def get_notices(self):
        """Return and clear the messages waiting for the GUI."""
        notices = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except queue.Empty:
                return notices

    def _run(self):
        """Open the camera, process frames until stopped, then release the camera."""
        try:
            self._camera = self.camera_factory(self.config.camera)
            # Only tags listed in the config are accepted, which rules out false detections.
            self._pipeline = TagPipeline(
                self.config.build_trackers(),
                self._camera.camera_params,
                self.config.tag_size_m,
                tag_family=self.config.tag_family,
                decision_margin=self.config.decision_margin,
                reference_id=self.config.reference_tag,
            )
            self._exposure_range = self._camera.exposure_range()
            if self._exposure is None:
                self._exposure = self._camera.current_exposure()
            while not self._stop_requested.is_set():
                self._apply_exposure_request()
                frame = self._camera.read()
                if frame is not None:
                    self._process_frame(frame)
        except Exception as error:  # Shown in the GUI instead of ending silently.
            self._notices.put(Notice(f"Camera stopped: {error}", is_error=True, is_fatal=True))
        finally:
            self._close_camera()

    def _apply_exposure_request(self):
        """Apply the newest exposure request from the capture thread."""
        with self._lock:
            request = self._exposure_request
            self._exposure_request = None
        if request is None:
            return

        auto, value = request
        value = self._exposure if value is None else value
        try:
            if auto:
                self._camera.set_auto_exposure(True)
            else:
                self._camera.set_exposure(value)
        except Exception as error:  # Tracking can continue at the old exposure.
            self._notices.put(Notice(f"Could not change exposure: {error}", is_error=True))
            return
        self._auto_exposure = auto
        self._exposure = value

    def _close_camera(self):
        """Release the camera, reporting any failure."""
        if self._camera is None:
            return
        try:
            self._camera.close()
        except Exception as error:
            self._notices.put(Notice(f"Could not close camera: {error}", is_error=True))

    def _process_frame(self, frame):
        """Detect tags, update any recording, and publish the display state."""
        timestamp = time.time()
        now = time.monotonic()
        tag_dict = self._pipeline.detect_tags(frame)

        with self._lock:
            reference_id = self._reference_id
            request = self._recording_request

        annotated = frame.copy()  # Camera frames are read-only.
        annotate.draw_tag_axes(
            annotated,
            tag_dict,
            self._camera.K,
            self._camera.dist_coeffs,
            reference_id=reference_id,
            names=self.names,
        )

        if request is not None:
            self._begin_recording(request, reference_id, tag_dict, annotated, timestamp, now)
        if self._recording is not None:
            self._add_to_recording(tag_dict, timestamp)

        remaining = None
        if self._recording is not None:
            remaining = max(self._recording.deadline - now, 0.0)
            if now >= self._recording.deadline:
                self._finish_recording()
                remaining = None

        self._publish(annotated, tag_dict, reference_id, remaining, now)

    def _publish(self, annotated, tag_dict, reference_id, remaining, now):
        """Store the latest display state."""
        self._seen_ids.update(tag_dict)
        self._frame_times.append(now)
        fps = 0.0
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

        if reference_id != self._filter_reference_id:
            self._position_filter.clear()
            self._filter_reference_id = reference_id
        if reference_id in tag_dict:
            for tag_id, tag in tag_dict.items():
                if tag_id != reference_id:
                    position, _ = relative_pose(tag_dict[reference_id], tag)
                    self._position_filter.add(tag_id, now, position)

        with self._lock:
            previous = self._state.sequence if self._state is not None else 0
            self._state = CaptureState(
                sequence=previous + 1,
                frame=annotated,
                fps=fps,
                visible_ids=frozenset(tag_dict),
                seen_ids=tuple(sorted(self._seen_ids)),
                relative_positions=self._position_filter.medians(now),
                reference_id=reference_id,
                recording_remaining=remaining,
                exposure_range=self._exposure_range,
                exposure=self._exposure,
                auto_exposure=self._auto_exposure,
            )

    def _begin_recording(self, request, reference_id, tag_dict, annotated, timestamp, now):
        """Start a recording, using this frame as the snapshot."""
        seconds, bush_id, cane_id = request
        recording = None
        if reference_id in tag_dict:
            recording = _Recording(
                bush_id=bush_id,
                cane_id=cane_id,
                seconds=seconds,
                reference_id=reference_id,
                started_at=timestamp,
                deadline=now + seconds,
                snapshot=annotated.copy(),
                collector=DataCollector(
                    metadata=self._build_metadata(reference_id, bush_id, cane_id, seconds)
                ),
            )
        else:
            self._notices.put(Notice("Reference tag is not visible, so nothing was recorded.", is_error=True))

        # Cleared together so a second request can't arrive between the two.
        with self._lock:
            self._recording = recording
            self._recording_request = None

    def _build_metadata(self, reference_id, bush_id, cane_id, seconds):
        """Return the Parquet metadata for a recording."""
        camera_config = self.config.camera
        return {
            "capture_start_timestamp": time.time(),
            "coordinate_frame": "camera",
            "reference_tag_id": reference_id,
            "bush_id": bush_id,
            "cane_id": cane_id,
            "requested_seconds": seconds,
            "position_unit": "m",
            "quaternion_order": "xyzw",
            "tag_family": self.config.tag_family,
            "tag_size_m": self.config.tag_size_m,
            "decision_margin_threshold": self.config.decision_margin,
            "tag_names": {str(tag_id): name for tag_id, name in self.names.items()},
            "camera": {
                "stream": "color",
                "width": camera_config.width,
                "height": camera_config.height,
                "fps": camera_config.fps,
                "manual_exposure": None if self._auto_exposure else self._exposure,
                "auto_exposure": self._auto_exposure,
                "intrinsics_matrix": self._camera.K.tolist(),
                "distortion_coefficients": self._camera.dist_coeffs.tolist(),
            },
        }

    def _add_to_recording(self, tag_dict, timestamp):
        """Add one frame's raw rows, and its poses relative to the reference."""
        recording = self._recording
        reference_tag = tag_dict.get(recording.reference_id)

        for tag_id, tag in tag_dict.items():
            recording.seen_ids.add(tag_id)
            x, y, z = tag.pose_t.reshape(3)
            quaternion = R.from_matrix(tag.pose_R).as_quat()
            recording.collector.update(
                timestamp, self.names.get(tag_id, f"tag_{tag_id}"), x, y, z, *quaternion
            )

            if reference_tag is not None and tag_id != recording.reference_id:
                position, rotation = relative_pose(reference_tag, tag)
                recording.positions.setdefault(tag_id, []).append(position)
                recording.rotations.setdefault(tag_id, []).append(rotation)

    def _finish_recording(self):
        """Average a finished recording, save its files, and report the result."""
        recording = self._recording
        try:
            averages = {
                tag_id: average_poses(recording.positions[tag_id], recording.rotations[tag_id])
                for tag_id in recording.positions
            }
            paths = build_output_paths(
                self.output_dir,
                recording.bush_id,
                recording.cane_id,
                datetime.fromtimestamp(recording.started_at),
            )
            write_average_csv(
                paths.csv,
                recording.bush_id,
                recording.cane_id,
                recording.started_at,
                recording.reference_id,
                averages,
                self.names,
            )
            recording.collector.dump(
                paths.parquet,
                metadata={
                    "capture_end_timestamp": time.time(),
                    "row_count": len(recording.collector.rows),
                },
            )
            if not cv2.imwrite(str(paths.snapshot), recording.snapshot):
                raise OSError(f"Could not write {paths.snapshot}")
        except Exception as error:  # Usually a full disk; the GUI must keep running.
            self._notices.put(Notice(f"Save failed: {error}", is_error=True))
        else:
            self._notices.put(Notice(f"Saved {paths.csv} and its frames and snapshot."))
            for warning in self._recording_warnings(recording, averages):
                self._notices.put(Notice(warning, is_error=True))
        finally:
            with self._lock:
                self._recording = None

    def _recording_warnings(self, recording, averages):
        """Return messages about tags with missing or unreliable data."""
        warnings = []
        if not averages:
            warnings.append("No tags were seen alongside the reference, so the CSV has no rows.")

        unmatched = recording.seen_ids - set(averages) - {recording.reference_id}
        if unmatched:
            warnings.append(f"Not in the CSV, never seen with the reference: {sorted(unmatched)}")

        missing = self._seen_ids - recording.seen_ids - {recording.reference_id}
        if missing:
            warnings.append(f"Not seen during this recording: {sorted(missing)}")

        for tag_id, average in sorted(averages.items()):
            total = average.n_frames + average.n_rejected
            if average.n_rejected > REJECTED_WARNING_FRACTION * total:
                warnings.append(
                    f"Tag {tag_id}: {average.n_rejected} of {total} frames dropped as flips."
                )
        return warnings
