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

CSV_COLUMNS = [
    "bush_id", "cane_id", "timestamp", "reference_id", "tag_id", "name",
    "x", "y", "z", "qx", "qy", "qz", "qw",
    "n_frames", "n_rejected", "pos_std_mm",
]


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


def relative_pose(reference_tag, tag):
    """Return a tag's position and rotation in the reference tag's frame."""
    reference_rotation = np.asarray(reference_tag.pose_R, dtype=np.float64)
    reference_position = np.asarray(reference_tag.pose_t, dtype=np.float64).reshape(3)
    tag_rotation = np.asarray(tag.pose_R, dtype=np.float64)
    tag_position = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)

    position = reference_rotation.T @ (tag_position - reference_position)
    rotation = reference_rotation.T @ tag_rotation
    return position, rotation


def _typical_rotation_angles(rotations):
    """Return each rotation's angle, in degrees, from the most central sample.

    The most central sample is used instead of the mean because flipped frames
    pull the mean towards them.
    """
    quaternions = rotations.as_quat()
    alignment = np.clip(np.abs(quaternions @ quaternions.T), 0.0, 1.0)
    pairwise_angles = 2.0 * np.arccos(alignment)
    central_index = np.argmin(np.median(pairwise_angles, axis=1))
    return np.degrees(pairwise_angles[central_index])


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
    keep = _typical_rotation_angles(all_rotations) <= ROTATION_OUTLIER_DEG

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


def build_output_paths(directory, bush_id, cane_id, started_at):
    """Return unused output paths for a recording, named by IDs and start time."""
    directory = Path(directory)
    stamp = started_at.strftime("%Y%m%d_%H%M")
    stem = f"bush{clean_id(bush_id)}_cane{clean_id(cane_id)}_{stamp}"

    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_{attempt}"
        paths = OutputPaths(
            csv=directory / f"{stem}{suffix}_average.csv",
            parquet=directory / f"{stem}{suffix}_frames.parquet",
            snapshot=directory / f"{stem}{suffix}_snapshot.png",
        )
        if not any(path.exists() for path in (paths.csv, paths.parquet, paths.snapshot)):
            return paths
        attempt += 1


def write_average_csv(path, bush_id, cane_id, timestamp, reference_id, averages, names):
    """Write one row per tag, in the reference tag's frame.

    Args:
        path:         Output CSV path.
        bush_id:      Bush ID.
        cane_id:      Cane ID.
        timestamp:    Recording start, as Unix seconds.
        reference_id: Reference tag ID.
        averages:     dict of tag_id -> AveragedPose.
        names:        dict of tag_id -> display name.
    """
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
                names.get(tag_id, f"tag_{tag_id}"),
                *(f"{value:.6f}" for value in average.position),
                *(f"{value:.6f}" for value in average.quaternion),
                average.n_frames,
                average.n_rejected,
                f"{average.pos_std_mm:.3f}",
            ])

