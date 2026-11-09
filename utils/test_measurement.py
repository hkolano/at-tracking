"""Tests for the GUI's pose math and recording, using a fake camera.

Run from the repository root:
    python -m pytest utils/test_measurement.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation as R

from src import capture
from src.capture import CaptureWorker
from src.config import CameraConfig, Config, TrackedObjectConfig
from src.data_collector import DataCollector
from src.gui import exposure_to_slider, slider_to_exposure
from src.measurement import RollingMedian, average_poses, build_output_paths, relative_pose

FOCAL = 900.0
WIDTH, HEIGHT = 1280, 720
TAG_PIXELS = 160
TAG_SIZE_M = 0.017
TAG_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


class FakePose:
    """Stands in for a detection with a pose."""

    def __init__(self, rotation, position):
        """Store a pose in the same shape as a detector result."""
        self.pose_R = np.asarray(rotation, dtype=np.float64)
        self.pose_t = np.asarray(position, dtype=np.float64).reshape(3, 1)


def render(tags):
    """Return a BGR image with tags at the given pixel centres."""
    image = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    half = TAG_PIXELS // 2
    for tag_id, (x, y) in tags.items():
        marker = cv2.aruco.generateImageMarker(TAG_DICTIONARY, tag_id, TAG_PIXELS)
        image[y - half:y + half, x - half:x + half] = marker[..., None]
    return image


class FakeCamera:
    """Serves a scripted sequence of frames, repeating the last one."""

    script = []
    later_scripts = []  # Scripts for cameras opened after a reconnect, in order.
    open_failures = 0   # Number of upcoming opens that fail, like an unplugged camera.
    exposure_error = None
    instances = []

    def __init__(self, config):
        """Set up intrinsics and start the script, or fail to open."""
        if FakeCamera.open_failures > 0:
            FakeCamera.open_failures -= 1
            raise RuntimeError("No device connected")
        self.config = config
        self.script = FakeCamera.later_scripts.pop(0) if FakeCamera.instances and FakeCamera.later_scripts else FakeCamera.script
        FakeCamera.instances.append(self)
        self.camera_params = (FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2)
        self.K = np.array(
            [[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        self.index = 0
        self.closed = False
        self.exposure_calls = []
        FakeCamera.instance = self

    def exposure_range(self):
        """Return the same range as the real camera."""
        return 1.0, 10000.0

    def current_exposure(self):
        """Return a fixed exposure setting."""
        return 166.0

    def set_exposure(self, value):
        """Log a manual exposure, or fail if told to."""
        if self.exposure_error is not None:
            raise self.exposure_error
        self.exposure_calls.append(("manual", value))

    def set_auto_exposure(self, enabled):
        """Log an auto exposure change."""
        self.exposure_calls.append(("auto", enabled))

    def read(self):
        """Return the next frame, paced like a slow camera."""
        time.sleep(0.01)
        tags = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        if isinstance(tags, Exception):
            raise tags
        return render(tags)

    def close(self):
        """Record that the camera was released."""
        self.closed = True


def make_config(exposure=100):
    """Return a config listing tags 0 to 3 as cane_0 to cane_3."""
    objects = tuple(
        TrackedObjectConfig(f"cane_{tag_id}", (tag_id,), ({"pos": np.zeros(3), "rot": np.eye(3)},))
        for tag_id in range(4)
    )
    return Config(tag_size_m=TAG_SIZE_M, objects=objects, camera=CameraConfig(exposure=exposure))


def wait_for(condition, timeout=20.0):
    """Wait until a condition is true, failing on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    pytest.fail("Timed out")


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Yield a worker factory, stopping whatever it starts."""
    started = []
    monkeypatch.setattr(capture, "RECONNECT_DELAY_SECONDS", 0.05)

    def make(script, reference=2, exposure=100, later_scripts=(), open_failures=0):
        """Start a worker on a scripted camera."""
        FakeCamera.script = script
        FakeCamera.later_scripts = list(later_scripts)
        FakeCamera.open_failures = open_failures
        FakeCamera.instances = []
        FakeCamera.exposure_error = None
        result = CaptureWorker(
            make_config(exposure), output_dir=tmp_path, camera_factory=FakeCamera
        )
        result.set_reference(reference)
        started.append(result)
        result.start()
        return result

    yield make
    for result in started:
        result.stop()


def single_file(directory, pattern):
    """Return the only file matching a pattern."""
    matches = list(Path(directory).glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def read_rows(path):
    """Return a CSV's rows as dicts."""
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_relative_pose_recovers_known_transform():
    """A tag placed by a known transform comes back as that transform."""
    reference_rotation = R.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
    reference_position = np.array([0.1, -0.2, 1.0])
    expected_rotation = R.from_euler("z", 30, degrees=True).as_matrix()
    expected_position = np.array([0.3, 0.05, -0.02])

    tag = FakePose(
        reference_rotation @ expected_rotation,
        reference_position + reference_rotation @ expected_position,
    )
    position, rotation = relative_pose(FakePose(reference_rotation, reference_position), tag)

    assert position == pytest.approx(expected_position)
    assert rotation == pytest.approx(expected_rotation)


def test_average_uses_all_frames_without_flips():
    """Small noise averages out and nothing is rejected."""
    rng = np.random.default_rng(0)
    positions = np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.001, (200, 3))
    rotations = R.from_euler("xyz", rng.normal(0, 1, (200, 3)), degrees=True).as_matrix()

    average = average_poses(positions, rotations)

    assert average.n_frames == 200 and average.n_rejected == 0
    assert average.position == pytest.approx([1, 2, 3], abs=5e-4)
    assert R.from_quat(average.quaternion).magnitude() < np.radians(0.5)
    assert 0.5 < average.pos_std_mm < 1.5


def test_average_drops_flipped_frames():
    """Flipped frames are dropped, along with their shifted positions."""
    rotations = [np.eye(3)] * 18 + [R.from_euler("x", 60, degrees=True).as_matrix()] * 2
    positions = [[0.0, 0.0, 1.0]] * 18 + [[0.5, 0.0, 1.0]] * 2

    average = average_poses(positions, rotations)

    assert average.n_frames == 18 and average.n_rejected == 2
    assert average.position == pytest.approx([0, 0, 1])
    assert R.from_quat(average.quaternion).magnitude() == pytest.approx(0, abs=1e-6)


def test_average_survives_even_split():
    """A 50/50 split keeps one group rather than rejecting every frame."""
    rotations = [np.eye(3)] * 5 + [R.from_euler("x", 90, degrees=True).as_matrix()] * 5
    average = average_poses([[0.0, 0.0, 1.0]] * 10, rotations)
    assert average.n_frames == 5 and average.n_rejected == 5


def test_average_of_no_frames_is_none():
    """No frames gives no pose."""
    assert average_poses([], []) is None


def test_output_paths_sanitize_and_never_collide(tmp_path):
    """Labels become safe file names and existing files aren't reused."""
    started = datetime(2026, 11, 6, 15, 4, 5)

    first = build_output_paths(tmp_path, "3", "12 b?", started)
    assert first.csv.name == "bush3_cane12_b_20261106_1504_average.csv"
    assert first.parquet.name == "bush3_cane12_b_20261106_1504_frames.parquet"
    assert first.snapshot.name == "bush3_cane12_b_20261106_1504_snapshot.png"

    first.csv.touch()
    second = build_output_paths(tmp_path, "3", "12 b?", started)
    assert second.csv.name == "bush3_cane12_b_20261106_1504_2_average.csv"


def test_recording_writes_all_three_files(worker, tmp_path):
    """A recording produces the CSV, Parquet, and snapshot described in the README."""
    live = worker([{0: (300, 200), 1: (900, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None and 2 in live.get_state().visible_ids)
    live.start_recording(0.5, "3", "12")
    wait_for(lambda: list(tmp_path.glob("*.png")))

    rows = read_rows(single_file(tmp_path, "bush3_cane12_*_average.csv"))
    assert [row["tag_id"] for row in rows] == ["0", "1"]
    assert [row["name"] for row in rows] == ["cane_0", "cane_1"]
    assert all(row["reference_id"] == "2" for row in rows)
    assert all(row["bush_id"] == "3" and row["cane_id"] == "12" for row in rows)
    assert all(int(row["n_frames"]) > 0 and row["n_rejected"] == "0" for row in rows)

    # Tags are 300 px apart horizontally at this fake camera's known depth.
    depth = FOCAL * TAG_SIZE_M / TAG_PIXELS
    expected = np.hypot(300, 300) * depth / FOCAL
    measured = np.linalg.norm([float(rows[0][axis]) for axis in "xyz"])
    assert measured == pytest.approx(expected, abs=1e-3)

    table = pq.read_table(single_file(tmp_path, "*_frames.parquet"))
    metadata = json.loads(table.schema.metadata[b"dataset_metadata"])
    assert metadata["coordinate_frame"] == "camera"
    assert metadata["reference_tag_id"] == 2
    assert metadata["bush_id"] == "3" and metadata["cane_id"] == "12"
    assert set(table.to_pandas()["name"]) == {"cane_0", "cane_1", "cane_2"}

    snapshot = cv2.imread(str(single_file(tmp_path, "*_snapshot.png")))
    assert snapshot.shape == (HEIGHT, WIDTH, 3)
    assert not np.array_equal(snapshot, render({0: (300, 200), 1: (900, 200), 2: (600, 500)}))


def test_record_without_visible_reference_saves_nothing(worker, tmp_path):
    """A request with the reference hidden is refused with a message."""
    live = worker([{0: (300, 200)}])
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(0.2, "1", "1")
    wait_for(lambda: not live.is_recording())

    assert not list(tmp_path.iterdir())
    assert any("not visible" in notice.text for notice in live.get_notices())


def test_second_request_during_recording_is_ignored(worker, tmp_path):
    """Extra clicks don't restart or overwrite a recording."""
    live = worker([{0: (300, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(0.4, "1", "1")
    for _ in range(5):
        live.start_recording(0.4, "1", "1")
        time.sleep(0.02)
    wait_for(lambda: list(tmp_path.glob("*.png")))
    wait_for(lambda: not live.is_recording())

    assert len(list(tmp_path.glob("*_average.csv"))) == 1


def test_reference_hidden_mid_recording_uses_shared_frames(worker, tmp_path):
    """Frames without the reference are raw-only and left out of the average."""
    both = {0: (300, 200), 2: (600, 500)}
    live = worker([both] * 40 + [{0: (300, 200)}] * 2000)
    wait_for(lambda: live.get_state() is not None and 2 in live.get_state().visible_ids)
    live.start_recording(4.0, "1", "1")
    wait_for(lambda: list(tmp_path.glob("*.png")), timeout=30)

    row = read_rows(single_file(tmp_path, "*_average.csv"))[0]
    frames = pq.read_table(single_file(tmp_path, "*_frames.parquet")).to_pandas()
    assert 0 < int(row["n_frames"]) < (frames["name"] == "cane_0").sum()


def test_tag_seen_earlier_but_not_recorded_is_reported(worker, tmp_path):
    """A tag that left the view is named in a warning, and has no CSV row."""
    live = worker([{0: (300, 200), 1: (900, 200), 2: (600, 500)}] * 3 + [{0: (300, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None and 1 not in live.get_state().visible_ids
             and 1 in live.get_state().seen_ids)
    live.start_recording(0.2, "1", "1")
    wait_for(lambda: list(tmp_path.glob("*.png")))
    wait_for(lambda: not live.is_recording())

    assert [row["tag_id"] for row in read_rows(single_file(tmp_path, "*_average.csv"))] == ["0"]
    assert any("Not seen during this recording: [1]" in n.text for n in live.get_notices())


def test_stop_during_recording_discards_and_releases_camera(worker, tmp_path):
    """Closing mid-recording writes nothing and closes the camera."""
    live = worker([{0: (300, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(60, "1", "1")
    wait_for(lambda: live.get_state().recording_remaining is not None)
    live.stop()

    assert not list(tmp_path.iterdir())
    assert FakeCamera.instances[-1].closed


def test_lost_camera_reconnects(worker):
    """A camera error is reported, the camera is reopened, and frames resume."""
    live = worker(
        [{0: (300, 200)}] * 3 + [RuntimeError("Frame didn't arrive")],
        later_scripts=[[{0: (300, 200), 2: (600, 500)}]],
    )
    wait_for(lambda: len(FakeCamera.instances) == 2 and live.get_state() is not None
             and 2 in live.get_state().visible_ids)

    texts = [notice.text for notice in live.get_notices()]
    assert any("Camera lost" in text and "Frame didn't arrive" in text for text in texts)
    assert "Camera reconnected." in texts
    assert FakeCamera.instances[0].closed and live.is_connected()


def test_reconnect_keeps_exposure(worker):
    """The reopened camera gets the exposure that was in use, not the config's."""
    live = worker([{0: (300, 200)}], later_scripts=[[{0: (300, 200)}]])
    wait_for(lambda: live.get_state() is not None)
    live.set_exposure(400)
    wait_for(lambda: live.get_state().exposure == 400)

    FakeCamera.instances[0].script = [RuntimeError("Frame didn't arrive")]
    FakeCamera.instances[0].index = 0
    wait_for(lambda: len(FakeCamera.instances) == 2 and live.is_connected())
    assert FakeCamera.instances[1].config.exposure == 400

    live.set_auto_exposure(True)
    wait_for(lambda: live.get_state().auto_exposure)
    FakeCamera.instances[1].script = [RuntimeError("Frame didn't arrive")]
    FakeCamera.instances[1].index = 0
    FakeCamera.later_scripts = [[{0: (300, 200)}]]
    wait_for(lambda: len(FakeCamera.instances) == 3 and live.is_connected())
    assert FakeCamera.instances[2].config.exposure is None


def test_recording_discarded_when_camera_lost(worker, tmp_path):
    """A recording cut short by a lost camera saves nothing, and recording can resume."""
    both = {0: (300, 200), 2: (600, 500)}
    live = worker([both], later_scripts=[[both]])
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(60, "1", "1")
    wait_for(lambda: live.get_state().recording_remaining is not None)

    FakeCamera.instances[0].script = [RuntimeError("Frame didn't arrive")]
    FakeCamera.instances[0].index = 0
    wait_for(lambda: not live.is_recording() and len(FakeCamera.instances) == 2)

    assert not list(tmp_path.iterdir())
    assert any("discarded" in notice.text for notice in live.get_notices())
    wait_for(live.is_connected)
    live.start_recording(0.2, "1", "1")
    wait_for(lambda: list(tmp_path.glob("*.png")))


def test_waits_for_missing_camera_and_reports_once(worker):
    """A camera that isn't plugged in yet is retried, with one message."""
    live = worker([{0: (300, 200)}], open_failures=5)
    wait_for(lambda: live.get_state() is not None)

    texts = [notice.text for notice in live.get_notices()]
    assert sum("Waiting for camera" in text for text in texts) == 1
    assert "Camera connected." in texts


def test_stop_while_waiting_for_camera_is_quick(worker, monkeypatch):
    """Closing the GUI doesn't wait out the retry delay."""
    monkeypatch.setattr(capture, "RECONNECT_DELAY_SECONDS", 30)
    live = worker([{0: (300, 200)}], open_failures=1000)
    wait_for(lambda: live.get_notices())
    started = time.monotonic()
    live.stop()
    assert time.monotonic() - started < 2 and not live._thread.is_alive()


def test_unchanged_exposure_sends_no_command(worker):
    """Repeating the current exposure, or auto when already auto, sends nothing."""
    live = worker([{0: (300, 200)}])
    wait_for(lambda: live.get_state() is not None)
    live.set_exposure(100)
    live.set_auto_exposure(False)
    current = live.get_state().sequence
    wait_for(lambda: live.get_state().sequence > current + 2)
    assert FakeCamera.instances[0].exposure_calls == []


def test_camera_skips_redundant_auto_exposure_command():
    """Changing manual exposure doesn't resend 'auto exposure off' each time."""
    import pyrealsense2 as rs
    from src.camera import RealSenseCamera

    class Sensor:
        """Records the options set on it."""

        def __init__(self):
            """Start with no calls."""
            self.calls = []

        def set_option(self, option, value):
            """Record one option change."""
            self.calls.append((option, value))

    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.color_sensor = Sensor()
    camera.auto_exposure_on = None

    camera.set_exposure(100)
    camera.set_exposure(200)
    camera.set_exposure(300)
    camera.set_auto_exposure(True)
    camera.set_exposure(400)

    auto_off = (rs.option.enable_auto_exposure, 0)
    assert camera.color_sensor.calls == [
        auto_off,
        (rs.option.exposure, 100),
        (rs.option.exposure, 200),
        (rs.option.exposure, 300),
        (rs.option.enable_auto_exposure, 1),
        auto_off,
        (rs.option.exposure, 400),
    ]


def test_save_failure_is_reported_and_worker_continues(worker, tmp_path):
    """A failed save is shown, and the camera keeps running."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a folder")
    live = worker([{0: (300, 200), 2: (600, 500)}])
    live.output_dir = blocker
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(0.2, "1", "1")
    wait_for(lambda: not live.is_recording())
    before = live.get_state().sequence

    wait_for(lambda: live.get_state().sequence > before)
    assert any("Save failed" in notice.text for notice in live.get_notices())


def test_recording_needs_both_ids(worker, tmp_path):
    """Empty or unusable IDs are refused."""
    live = worker([{0: (300, 200), 2: (600, 500)}])
    for bush_id, cane_id in (("", "1"), ("1", ""), ("  ", "1"), ("1", "?!")):
        with pytest.raises(ValueError):
            live.start_recording(1, bush_id, cane_id)
    assert not live.is_recording()


def test_rolling_median_ignores_single_outlier():
    """One wild frame doesn't move the median."""
    rolling = RollingMedian(window_seconds=1.0)
    for index, x in enumerate([0.10, 0.10, 0.50, 0.10, 0.10]):
        rolling.add(1, index * 0.1, np.array([x, 0.0, 0.0]))
    assert rolling.medians(0.4)[1] == pytest.approx([0.10, 0, 0])


def test_rolling_median_ages_out_and_clears():
    """Old samples leave the window, and clear() forgets everything."""
    rolling = RollingMedian(window_seconds=1.0)
    rolling.add(1, 0.0, np.array([1.0, 0.0, 0.0]))
    rolling.add(1, 0.9, np.array([3.0, 0.0, 0.0]))
    rolling.add(2, 0.9, np.array([5.0, 0.0, 0.0]))

    assert rolling.medians(1.0)[1] == pytest.approx([2, 0, 0])
    assert rolling.medians(1.5)[1] == pytest.approx([3, 0, 0])
    assert rolling.medians(2.5) == {}

    rolling.add(1, 3.0, np.array([1.0, 0.0, 0.0]))
    rolling.clear()
    assert rolling.medians(3.0) == {}


def test_display_is_filtered_but_recorded_rows_stay_raw():
    """A flipped frame is absent from the display and kept in the Parquet rows."""
    live = CaptureWorker(make_config(), camera_factory=FakeCamera)
    FakeCamera.instances = []
    live._camera = FakeCamera(None)
    reference = FakePose(np.eye(3), [0, 0, 1])
    live._recording = type("Recording", (), {})()
    live._recording.reference_id = 2
    live._recording.seen_ids = set()
    live._recording.positions = {}
    live._recording.rotations = {}
    live._recording.collector = DataCollector()

    for index, x in enumerate([0.1, 0.1, 0.9, 0.1, 0.1]):
        tags = {2: reference, 1: FakePose(np.eye(3), [x, 0, 1])}
        live._publish(np.zeros((2, 2, 3), np.uint8), tags, 2, None, 100.0 + index * 0.1)
        live._add_to_recording(tags, 100.0)

    assert live.get_state().relative_positions[1] == pytest.approx([0.1, 0, 0])
    assert max(row[2] for row in live._recording.collector.rows) == pytest.approx(0.9)
    assert max(position[0] for position in live._recording.positions[1]) == pytest.approx(0.9)


def test_changing_reference_clears_filter():
    """Positions from the old reference don't carry over."""
    live = CaptureWorker(make_config(), camera_factory=FakeCamera)
    tags = {
        1: FakePose(np.eye(3), [0.5, 0, 1]),
        2: FakePose(np.eye(3), [0, 0, 1]),
        3: FakePose(np.eye(3), [0.1, 0, 1]),
    }
    image = np.zeros((2, 2, 3), np.uint8)
    live._publish(image, tags, 2, None, 100.0)
    assert set(live.get_state().relative_positions) == {1, 3}

    live._publish(image, {3: tags[3]}, 3, None, 100.1)
    assert live.get_state().relative_positions == {}


def test_hidden_tag_keeps_value_for_one_second(worker):
    """A hidden tag stays displayed briefly, then disappears."""
    both = {0: (300, 200), 1: (900, 200), 2: (600, 500)}
    live = worker([both] * 5 + [{0: (300, 200), 2: (600, 500)}])

    def tag_1_hidden():
        """Return True once tag 1 has been seen and has left the view."""
        state = live.get_state()
        return state is not None and 1 in state.seen_ids and 1 not in state.visible_ids

    wait_for(tag_1_hidden)
    hidden_at = time.monotonic()
    assert 1 in live.get_state().relative_positions

    wait_for(lambda: 1 not in live.get_state().relative_positions)
    assert 0.5 < time.monotonic() - hidden_at < 1.5
    assert 0 in live.get_state().relative_positions


def test_exposure_requests_apply_on_worker(worker):
    """Only the newest manual value is applied, and the state reports it."""
    live = worker([{0: (300, 200)}])
    wait_for(lambda: live.get_state() is not None)
    assert live.get_state().exposure_range == (1.0, 10000.0)
    assert live.get_state().exposure == 100 and not live.get_state().auto_exposure

    live.set_exposure(200)
    live.set_exposure(300)
    wait_for(lambda: live.get_state().exposure == 300)
    assert FakeCamera.instances[-1].exposure_calls == [("manual", 300)]

    live.set_auto_exposure(True)
    wait_for(lambda: live.get_state().auto_exposure)
    live.set_auto_exposure(False)
    wait_for(lambda: not live.get_state().auto_exposure)
    assert live.get_state().exposure == 300
    assert FakeCamera.instances[-1].exposure_calls[-1] == ("manual", 300)


def test_exposure_ignored_during_recording(worker, tmp_path):
    """A recording keeps one exposure."""
    live = worker([{0: (300, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None)
    live.start_recording(0.3, "1", "1")
    live.set_exposure(500)
    wait_for(lambda: list(tmp_path.glob("*.png")))
    wait_for(lambda: not live.is_recording())
    current = live.get_state().sequence
    wait_for(lambda: live.get_state().sequence > current + 2)

    assert live.get_state().exposure == 100
    assert FakeCamera.instances[-1].exposure_calls == []


def test_exposure_failure_reopens_camera(worker):
    """A rejected exposure change reopens the camera at the last exposure that worked."""
    live = worker([{0: (300, 200)}], later_scripts=[[{0: (300, 200)}]])
    wait_for(lambda: live.get_state() is not None)
    FakeCamera.instances[0].exposure_error = RuntimeError("option rejected")
    live.set_exposure(500)
    wait_for(lambda: len(FakeCamera.instances) == 2 and live.is_connected())

    assert any("option rejected" in n.text and "Reconnecting" in n.text for n in live.get_notices())
    assert FakeCamera.instances[1].config.exposure == 100
    current = live.get_state().sequence
    wait_for(lambda: live.get_state().sequence > current)
    assert live.get_state().exposure == 100


def test_auto_config_reports_camera_exposure(worker):
    """With exposure: null, the slider starts from the camera's own value."""
    live = worker([{0: (300, 200)}], exposure=None)
    wait_for(lambda: live.get_state() is not None)
    assert live.get_state().auto_exposure and live.get_state().exposure == 166.0


def test_metadata_records_exposure_in_use(worker, tmp_path):
    """Metadata shows the slider's value, not the config's."""
    live = worker([{0: (300, 200), 2: (600, 500)}])
    wait_for(lambda: live.get_state() is not None)
    live.set_exposure(350)
    wait_for(lambda: live.get_state().exposure == 350)
    live.start_recording(0.2, "1", "1")
    wait_for(lambda: list(tmp_path.glob("*.png")))
    wait_for(lambda: not live.is_recording())

    table = pq.read_table(single_file(tmp_path, "*_frames.parquet"))
    camera = json.loads(table.schema.metadata[b"dataset_metadata"])["camera"]
    assert camera["manual_exposure"] == 350 and camera["auto_exposure"] is False


@pytest.mark.parametrize("exposure", [1, 7, 100, 166, 1234, 10000])
def test_exposure_slider_round_trip(exposure):
    """Slider positions map back to the same exposure."""
    position = exposure_to_slider(exposure, 1.0, 10000.0)
    assert slider_to_exposure(position, 1.0, 10000.0) == exposure


def test_exposure_slider_ends_match_camera_range():
    """The slider's ends are the camera's limits."""
    assert slider_to_exposure(0.0, 1.0, 10000.0) == 1
    assert slider_to_exposure(1.0, 1.0, 10000.0) == 10000
    assert exposure_to_slider(20000, 1.0, 10000.0) == 1.0


def test_tags_not_in_config_are_ignored(worker, tmp_path):
    """A false detection never reaches the display or the saved files."""
    live = worker([{0: (300, 200), 2: (600, 500), 5: (900, 200)}])
    wait_for(lambda: live.get_state() is not None)
    assert 5 not in live.get_state().seen_ids

    live.start_recording(0.2, "1", "1")
    wait_for(lambda: list(tmp_path.glob("*.png")))
    assert read_rows(single_file(tmp_path, "*_average.csv"))[0]["tag_id"] == "0"
    frames = pq.read_table(single_file(tmp_path, "*_frames.parquet")).to_pandas()
    assert set(frames["name"]) == {"cane_0", "cane_2"}


def test_stop_reports_whether_camera_was_released(worker):
    """stop() says whether the camera thread finished in time."""
    healthy = worker([{0: (300, 200)}])
    wait_for(lambda: healthy.get_state() is not None)
    assert healthy.stop() is True

    stuck = worker([{0: (300, 200)}])
    wait_for(lambda: stuck.get_state() is not None)
    FakeCamera.instances[-1].read = lambda: time.sleep(5)  # A camera that stops answering.
    time.sleep(0.1)
    started = time.monotonic()
    assert stuck.stop(timeout=0.3) is False
    assert time.monotonic() - started < 1


CLOSE_SCRIPT = """
import sys, threading, time
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
import tkinter as tk
import test_measurement as t
from src import gui
from src.capture import CaptureWorker

gui.CAMERA_RELEASE_SECONDS = 0.5
t.FakeCamera.script = [{{0: (300, 200)}}]
if {stuck}:
    t.FakeCamera.read = lambda self: time.sleep(1000)  # The camera stops answering.
worker = CaptureWorker(t.make_config(), camera_factory=t.FakeCamera)
root = tk.Tk()
app = gui.TrackerApp(root, worker)
worker.start()
root.after(1500, app.close)
root.mainloop()
# Without a forced exit, this thread would keep the process alive, as a stuck camera does.
threading.Thread(target=time.sleep, args=(1000,)).start()
gui.exit_if_camera_stuck(app)
print("closed normally", flush=True)
"""


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display for the window")
@pytest.mark.parametrize("stuck", [False, True])
def test_closing_window_with_stuck_camera_ends_program(stuck, tmp_path):
    """Closing the window ends the program even when the camera never answers."""
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "close.py"
    script.write_text(CLOSE_SCRIPT.format(root=str(root), tests=str(root / "utils"), stuck=stuck))

    process = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if stuck:
        _, errors = process.communicate(timeout=20)
        assert process.returncode == 1
        assert "unplug the camera" in errors
    else:
        # A healthy camera closes normally; the extra thread then keeps it open, so stop it.
        time.sleep(4)
        assert process.poll() is None
        process.kill()
        output, _ = process.communicate()
        assert "closed normally" in output


def test_camera_detects_its_own_reset():
    """A frame counter that starts again from zero means the camera reset itself."""
    from src.camera import CameraResetError, RealSenseCamera

    class Frame:
        """A color frame with a frame number."""

        def __init__(self, number):
            """Store the frame number."""
            self.number = number

        def __bool__(self):
            """Behave like a valid frame."""
            return True

        def get_frame_number(self):
            """Return the frame number."""
            return self.number

        def get_data(self):
            """Return a tiny image."""
            return np.zeros((2, 2, 3), np.uint8)

    class Pipeline:
        """Serves frames with the given numbers."""

        def __init__(self, numbers):
            """Store the frame numbers to serve."""
            self.numbers = iter(numbers)

        def wait_for_frames(self):
            """Return the next frame set."""
            frame = Frame(next(self.numbers))
            return type("Frames", (), {"get_color_frame": lambda _self: frame})()

    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.pipeline = Pipeline([5, 6, 7, 0])
    camera.last_frame_number = None
    for _ in range(3):
        camera.read()
    with pytest.raises(CameraResetError):
        camera.read()
