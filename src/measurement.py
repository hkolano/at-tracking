"""Pose math and output files for one measurement. No camera or GUI code."""

import csv
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Length of the median filter on the live readout.
DISPLAY_FILTER_SECONDS = 1.0

# Frames further than this from a tag's typical rotation count as orientation flips.
ROTATION_OUTLIER_DEG = 15.0
# The typical rotation is chosen from at most this many evenly spaced samples, so long
# recordings don't need an all-pairs comparison (which grows with the square).
MAX_CENTRAL_CANDIDATES = 200

# Recording file names: <stem>_<YYYYmmdd>_<HHMM>[_n]<suffix>.
AVERAGE_SUFFIX = "_average.csv"
FRAMES_SUFFIX = "_frames.parquet"
SNAPSHOT_SUFFIX = "_snapshot.png"

CSV_COLUMNS = [
    "bush_id", "cane_id", "timestamp", "reference_id", "tag_id", "name",
    "x", "y", "z", "qx", "qy", "qz", "qw",
    "n_frames", "n_rejected", "pos_std_mm",
    "detected_pct", "flipped_pct", "jitter_mm", "well_detected",
]

STATUS_GOOD = "good"
STATUS_FAIR = "fair"
STATUS_NONE = "none"


@dataclass(frozen=True)
class AveragedPose:
    """One tag's averaged pose over a recording."""

    position: np.ndarray    # (3,) metres
    quaternion: np.ndarray  # (4,) xyzw
    n_frames: int           # frames used in the average
    n_rejected: int         # frames dropped as orientation flips
    pos_std_mm: float       # largest per-axis standard deviation


@dataclass(frozen=True)
class OutputPaths:
    """Files written for one recording."""

    csv: Path
    parquet: Path
    snapshot: Path


class RollingMedian:
    """Median position per tag over a recent time window, for display."""

    def __init__(self, window_seconds=DISPLAY_FILTER_SECONDS):
        """Start with no samples."""
        self.window_seconds = window_seconds
        self.samples = {}  # tag_id -> deque of (time, position)

    def add(self, tag_id, sample_time, position):
        """Add one position for a tag."""
        self.samples.setdefault(tag_id, deque()).append((sample_time, position))

    def medians(self, now):
        """Return tag_id -> median position, for tags with samples inside the window."""
        results = {}
        for tag_id in list(self.samples):
            tag_samples = self.samples[tag_id]
            while tag_samples and now - tag_samples[0][0] > self.window_seconds:
                tag_samples.popleft()
            if tag_samples:
                results[tag_id] = np.median([position for _, position in tag_samples], axis=0)
            else:
                del self.samples[tag_id]
        return results

    def clear(self):
        """Forget all samples."""
        self.samples.clear()


@dataclass(frozen=True)
class TagQuality:
    """How well one tag has been detected over a recent window, in the camera frame."""

    detected_fraction: float  # share of frames in the window where the tag was found
    flip_fraction: float      # share of those detections that are orientation flips
    jitter_mm: float | None   # robust spread of position on the worst axis; None if too few frames
    seconds_since_seen: float


class DetectionQuality:
    """Measure how consistently each tag is detected, frame by frame."""

    # Below this many unflipped detections, jitter isn't measured.
    MIN_FRAMES_FOR_JITTER = 3

    def __init__(self, window_seconds):
        """Start with no frames."""
        self.window_seconds = window_seconds
        self.frame_times = deque()
        self.samples = {}  # tag_id -> deque of (time, position, quaternion xyzw)

    def add_frame(self, now, detections):
        """Add one frame's detections.

        Args:
            now:        Frame time in seconds.
            detections: dict of tag_id -> detection with pose_t and pose_R, in the
                        camera frame. Tags missing from it count as not detected.
        """
        self.frame_times.append(now)
        if not detections:
            return
        # One conversion per frame, rather than one per sample every time it's judged.
        quaternions = R.from_matrix(
            np.array([detection.pose_R for detection in detections.values()], dtype=np.float64)
        ).as_quat()
        for (tag_id, detection), quaternion in zip(detections.items(), quaternions):
            position = np.asarray(detection.pose_t, dtype=np.float64).reshape(3)
            self.samples.setdefault(tag_id, deque()).append((now, position, quaternion))

    def qualities(self, now):
        """Return tag_id -> TagQuality for tags detected within the window."""
        while self.frame_times and now - self.frame_times[0] > self.window_seconds:
            self.frame_times.popleft()
        results = {}
        for tag_id in list(self.samples):
            tag_samples = self.samples[tag_id]
            while tag_samples and now - tag_samples[0][0] > self.window_seconds:
                tag_samples.popleft()
            if not tag_samples:
                del self.samples[tag_id]
                continue
            results[tag_id] = self._quality(tag_samples, now)
        return results

    def _quality(self, tag_samples, now):
        """Summarize one tag's detections in the window."""
        positions = np.array([position for _, position, _ in tag_samples])
        quaternions = np.array([quaternion for _, _, quaternion in tag_samples])
        flipped = _typical_rotation_angles(quaternions) > ROTATION_OUTLIER_DEG

        kept = positions[~flipped]
        jitter_mm = None
        if len(kept) >= self.MIN_FRAMES_FOR_JITTER:
            # Median absolute deviation, scaled to match a standard deviation for
            # normal noise, so a single bad frame can't dominate.
            spread = 1.4826 * np.median(np.abs(kept - np.median(kept, axis=0)), axis=0)
            jitter_mm = float(spread.max() * 1000.0)

        return TagQuality(
            detected_fraction=min(len(tag_samples) / max(len(self.frame_times), 1), 1.0),
            flip_fraction=float(flipped.mean()),
            jitter_mm=jitter_mm,
            seconds_since_seen=now - tag_samples[-1][0],
        )


@dataclass(frozen=True)
class QualityLimits:
    """Limits for judging a tag's detection.

    The GUI sets these from the constants at the top of gui.py; the defaults here
    match them, for code that runs without the GUI.
    """

    min_detected: float = 0.20         # below this share of frames: not detected
    green_min_detected: float = 0.70   # share of frames
    green_max_flipped: float = 0.15    # share of detections
    green_max_jitter_mm: float = 10.0
    not_seen_seconds: float = 1.0      # live view only; see judge_quality


def judge_quality(quality, limits):
    """Judge a tag's live detection: like judge_detection, but not seen recently is none."""
    if quality is not None and quality.seconds_since_seen > limits.not_seen_seconds:
        return STATUS_NONE, ""
    return judge_detection(quality, limits)


def judge_detection(quality, limits):
    """Return (status, text) for a TagQuality, or for None if the tag isn't detected.

    How recently the tag was seen isn't considered, so this also judges whole
    recordings. The text is short enough for the GUI table: the jitter when the tag
    is good, or what keeps it from being good.
    """
    if quality is None or quality.detected_fraction < limits.min_detected:
        return STATUS_NONE, ""

    problems = []
    if quality.detected_fraction < limits.green_min_detected:
        problems.append(f"seen {quality.detected_fraction:.0%}")
    if quality.flip_fraction > limits.green_max_flipped:
        problems.append(f"flips {quality.flip_fraction:.0%}")
    if quality.jitter_mm is None:
        problems.append("few frames")
    elif quality.jitter_mm > limits.green_max_jitter_mm:
        problems.append(f"jitter {quality.jitter_mm:.0f} mm")

    if problems:
        return STATUS_FAIR, ", ".join(problems)
    return STATUS_GOOD, f"good, {quality.jitter_mm:.0f} mm"


@dataclass(frozen=True)
class RecordingVerdict:
    """Which tags a recording captured well enough to use."""

    well_detected: frozenset  # tags green over the recording, with a green reference
    problems: dict            # tag_id -> why it wasn't good enough, for tags in the CSV
    reference_problem: str    # why the reference wasn't good enough, or ""


def judge_recording(quality, reference_id, averaged_ids, limits):
    """Decide which tags in a recording's CSV were well detected.

    Each tag is judged over the whole recording, so how recently it was seen
    doesn't matter. Nothing counts if the reference itself wasn't green.

    Args:
        quality:      dict of tag_id -> TagQuality over the whole recording.
        reference_id: Reference tag ID.
        averaged_ids: Tags with a row in the CSV.
        limits:       QualityLimits.
    """
    def judge(tag_id):
        """Judge one tag over the recording."""
        status, text = judge_detection(quality.get(tag_id), limits)
        return status, text or "rarely seen"

    reference_status, reference_text = judge(reference_id)
    if reference_status != STATUS_GOOD:
        return RecordingVerdict(frozenset(), {}, reference_text)

    good, problems = set(), {}
    for tag_id in sorted(averaged_ids):
        status, text = judge(tag_id)
        if status == STATUS_GOOD:
            good.add(tag_id)
        else:
            problems[tag_id] = text
    return RecordingVerdict(frozenset(good), problems, "")


def quality_columns(quality, well_detected):
    """Return the CSV's detected_pct, flipped_pct, jitter_mm and well_detected values."""
    if quality is None:
        return ["", "", "", well_detected]
    jitter = "" if quality.jitter_mm is None else f"{quality.jitter_mm:.2f}"
    return [
        f"{quality.detected_fraction * 100:.1f}",
        f"{quality.flip_fraction * 100:.1f}",
        jitter,
        well_detected,
    ]


def relative_transform(reference_rotation, reference_position, rotation, position):
    """Return a pose's position and rotation in a reference pose's frame.

    Works on one pose, with (3, 3) rotations and (3,) positions, or on stacks of
    them with matching leading axes.
    """
    reference_inverse = np.swapaxes(reference_rotation, -1, -2)
    relative_position = (reference_inverse @ (position - reference_position)[..., None])[..., 0]
    return relative_position, reference_inverse @ rotation


def relative_pose(reference_tag, tag):
    """Return a tag's position and rotation in the reference tag's frame."""
    return relative_transform(
        np.asarray(reference_tag.pose_R, dtype=np.float64),
        np.asarray(reference_tag.pose_t, dtype=np.float64).reshape(3),
        np.asarray(tag.pose_R, dtype=np.float64),
        np.asarray(tag.pose_t, dtype=np.float64).reshape(3),
    )


def _typical_rotation_angles(quaternions):
    """Return each rotation's angle, in degrees, from the most central sample.

    The most central sample is used instead of the mean because flipped frames
    pull the mean towards them. It's chosen from at most MAX_CENTRAL_CANDIDATES
    evenly spaced samples; with fewer samples than that, every one is considered.

    Args:
        quaternions: (N, 4) xyzw quaternions.
    """
    count = len(quaternions)
    candidates = quaternions
    if count > MAX_CENTRAL_CANDIDATES:
        candidates = quaternions[np.linspace(0, count - 1, MAX_CENTRAL_CANDIDATES).round().astype(int)]
    alignment = np.clip(np.abs(candidates @ quaternions.T), 0.0, 1.0)
    angles = 2.0 * np.arccos(alignment)
    central = np.argmin(np.median(angles, axis=1))
    return np.degrees(angles[central])


def average_poses(positions, rotations):
    """Average a tag's per-frame poses after dropping orientation flips.

    Args:
        positions: Sequence of (3,) positions.
        rotations: Sequence of (3, 3) rotation matrices.

    Returns:
        An AveragedPose, or None if there are no frames.
    """
    if len(positions) == 0:
        return None

    positions = np.asarray(positions, dtype=np.float64)
    all_rotations = R.from_matrix(np.asarray(rotations, dtype=np.float64))
    keep = _typical_rotation_angles(all_rotations.as_quat()) <= ROTATION_OUTLIER_DEG

    kept_positions = positions[keep]
    return AveragedPose(
        position=kept_positions.mean(axis=0),
        quaternion=all_rotations[keep].mean().as_quat(),
        n_frames=int(keep.sum()),
        n_rejected=int((~keep).sum()),
        pos_std_mm=float(kept_positions.std(axis=0).max() * 1000.0),
    )


def clean_id(value):
    """Return a bush or cane ID that is safe for file names, possibly empty."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def recording_stem(bush_id, cane_id):
    """Return the start of every file name for one bush and cane."""
    return f"bush{clean_id(str(bush_id))}_cane{clean_id(str(cane_id))}"


def tag_name(names, tag_id):
    """Return a tag's display name, or tag_<id> if it has none."""
    return names.get(tag_id, f"tag_{tag_id}")


def row_pose(row):
    """Return the (3,) position and (4,) xyzw quaternion from an average CSV row."""
    position = np.array([float(row[axis]) for axis in ("x", "y", "z")])
    quaternion = np.array([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    return position, quaternion


def build_output_paths(directory, bush_id, cane_id, started_at):
    """Return unused output paths for a recording, named by IDs and start time."""
    directory = Path(directory)
    stem = f"{recording_stem(bush_id, cane_id)}_{started_at:%Y%m%d_%H%M}"

    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_{attempt}"
        paths = OutputPaths(
            csv=directory / f"{stem}{suffix}{AVERAGE_SUFFIX}",
            parquet=directory / f"{stem}{suffix}{FRAMES_SUFFIX}",
            snapshot=directory / f"{stem}{suffix}{SNAPSHOT_SUFFIX}",
        )
        if not any(path.exists() for path in (paths.csv, paths.parquet, paths.snapshot)):
            return paths
        attempt += 1


def write_average_csv(
    path, bush_id, cane_id, timestamp, reference_id, averages, names, detection=None
):
    """Write one row per tag, in the reference tag's frame.

    Args:
        path:         Output CSV path.
        bush_id:      Bush ID.
        cane_id:      Cane ID.
        timestamp:    Recording start, as Unix seconds.
        reference_id: Reference tag ID.
        averages:     dict of tag_id -> AveragedPose.
        names:        dict of tag_id -> display name.
        detection:    Optional dict of tag_id -> the four detection quality values
                      from quality_columns(). Missing tags get blank values.
    """
    detection = detection or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for tag_id in sorted(averages):
            average = averages[tag_id]
            writer.writerow([
                bush_id,
                cane_id,
                f"{timestamp:.3f}",
                reference_id,
                tag_id,
                tag_name(names, tag_id),
                *(f"{value:.6f}" for value in average.position),
                *(f"{value:.6f}" for value in average.quaternion),
                average.n_frames,
                average.n_rejected,
                f"{average.pos_std_mm:.3f}",
                *detection.get(tag_id, ["", "", "", ""]),
            ])

