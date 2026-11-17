"""Correct recordings made with the wrong size for one tag.

A tag's position scales exactly with its size, and its rotation doesn't depend on
size, so a wrong size can be undone by rescaling that tag's camera-frame positions.
This script does that in each recording's *_frames.parquet, then rebuilds the
matching *_average.csv from the corrected frames. The originals are first copied to
data/before_tag_size_fix/. Snapshots are left as they are.

Recordings that store detection quality have the corrected tag's jitter rescaled,
and are re-judged for well_detected with the limits saved in the recording.

Run from the repository root, for example:
    python -m utils.fix_tag_size --tag 0 --recorded-size 0.08 --true-size 0.0254 \
        data/bushctyd_cane*_frames.parquet
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data_collector import read_tracking_parquet, write_tracking_parquet
from src.measurement import (
    AVERAGE_SUFFIX,
    FRAMES_SUFFIX,
    QualityLimits,
    TagQuality,
    average_poses,
    judge_recording,
    quality_columns,
    relative_transform,
    row_pose,
    tag_name,
    write_average_csv,
)

BACKUP_DIR_NAME = "before_tag_size_fix"
POSITION_TOLERANCE_M = 1e-5  # CSVs store 6 decimals, so 1e-6 m rounding.
QUATERNION_TOLERANCE = 1e-4


def average_relative_poses(frames, reference_name, tag_ids):
    """Rebuild the CSV averages: each tag's pose relative to the reference, per frame.

    Args:
        frames:         Camera-frame rows, one per tag per frame.
        reference_name: Name of the reference tag's rows.
        tag_ids:        dict of name -> tag ID.

    Returns:
        dict of tag_id -> AveragedPose, as the GUI computes it.
    """
    columns = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    reference = frames.loc[frames["name"] == reference_name, ["timestamp", *columns]]
    reference = reference.drop_duplicates("timestamp")
    others = frames[frames["name"] != reference_name]
    # Each tag row next to the reference row from the same frame, in time order.
    paired = others.merge(reference, on="timestamp", suffixes=("", "_ref"))
    paired = paired.sort_values("timestamp", kind="stable")
    if paired.empty:
        return {}

    def pose_arrays(suffix):
        """Return stacked rotation matrices and positions for one side of the pairs."""
        quaternions = paired[[f"{key}{suffix}" for key in ("qx", "qy", "qz", "qw")]].to_numpy(float)
        positions = paired[[f"{axis}{suffix}" for axis in ("x", "y", "z")]].to_numpy(float)
        return R.from_quat(quaternions).as_matrix(), positions

    reference_rotations, reference_positions = pose_arrays("_ref")
    rotations, positions = pose_arrays("")
    relative_positions, relative_rotations = relative_transform(
        reference_rotations, reference_positions, rotations, positions
    )
    names = paired["name"].to_numpy()
    return {
        tag_ids[name]: average_poses(relative_positions[names == name], relative_rotations[names == name])
        for name in dict.fromkeys(names)
    }


def rescale_detection(metadata, tag_id, scale, reference_id, averaged_ids):
    """Rescale a tag's jitter in the metadata and re-judge well_detected.

    Updates metadata in place. Returns the CSV's detection values, or None for
    recordings made before detection quality was saved.
    """
    stored = metadata.get("detection_quality")
    if stored is None:
        return None
    entry = stored.get(str(tag_id))
    if entry is not None and entry["jitter_mm"] is not None:
        entry["jitter_mm"] *= scale  # Position spread scales with tag size too.

    quality = {
        int(key): TagQuality(
            value["detected_fraction"], value["flip_fraction"], value["jitter_mm"], 0.0
        )
        for key, value in stored.items()
    }
    limits = QualityLimits(**metadata["quality_limits"])
    verdict = judge_recording(quality, reference_id, averaged_ids, limits)
    for key, value in stored.items():
        value["well_detected"] = int(key) in verdict.well_detected
    metadata["well_detected_tag_ids"] = sorted(verdict.well_detected)
    metadata["reference_problem"] = verdict.reference_problem
    return {
        tag: quality_columns(quality.get(tag), tag in verdict.well_detected)
        for tag in averaged_ids
    }


def read_csv_rows(csv_path):
    """Return the rows of an average CSV as dicts keyed by tag ID."""
    with open(csv_path, newline="", encoding="utf-8") as file:
        return {int(row["tag_id"]): row for row in csv.DictReader(file)}


def matches_csv(averages, csv_rows):
    """Return True if rebuilt averages agree with an existing CSV."""
    if set(averages) != set(csv_rows):
        return False
    for tag_id, average in averages.items():
        row = csv_rows[tag_id]
        position, quaternion = row_pose(row)
        if np.abs(position - average.position).max() > POSITION_TOLERANCE_M:
            return False
        # q and -q are the same rotation.
        if min(np.abs(quaternion - average.quaternion).max(),
               np.abs(quaternion + average.quaternion).max()) > QUATERNION_TOLERANCE:
            return False
        if int(row["n_frames"]) != average.n_frames or int(row["n_rejected"]) != average.n_rejected:
            return False
    return True


def fix_recording(parquet_path, tag_id, recorded_size, true_size, dry_run=False):
    """Rescale one tag's positions in a recording and rebuild its average CSV."""
    parquet_path = Path(parquet_path)
    csv_path = parquet_path.with_name(parquet_path.name.replace(FRAMES_SUFFIX, AVERAGE_SUFFIX))
    frames, metadata = read_tracking_parquet(parquet_path)
    csv_rows = read_csv_rows(csv_path)

    recorded_sizes = metadata.get("tag_sizes_m") or {}
    size_in_file = recorded_sizes.get(str(tag_id), metadata["tag_size_m"])
    if not np.isclose(size_in_file, recorded_size):
        raise SystemExit(
            f"{parquet_path}: tag {tag_id} was recorded at {size_in_file} m, not "
            f"{recorded_size} m. Nothing changed. Already fixed?"
        )

    names_in_file = {int(key): name for key, name in metadata["tag_names"].items()}
    tag_ids = {name: key for key, name in names_in_file.items()}
    fixed_name = tag_name(names_in_file, tag_id)
    tag_ids.setdefault(fixed_name, tag_id)
    reference_id = metadata["reference_tag_id"]
    reference_name = tag_name(names_in_file, reference_id)

    # Check the averaging is rebuilt exactly as the GUI did it before changing anything.
    if not matches_csv(average_relative_poses(frames, reference_name, tag_ids), csv_rows):
        raise SystemExit(f"{csv_path}: couldn't reproduce the existing CSV. Nothing changed.")

    scale = true_size / recorded_size
    is_tag = frames["name"] == fixed_name
    frames.loc[is_tag, ["x", "y", "z"]] *= scale
    averages = average_relative_poses(frames, reference_name, tag_ids)
    detection = rescale_detection(metadata, tag_id, scale, reference_id, set(averages))

    print(f"{parquet_path.name}: {int(is_tag.sum())} rows of {fixed_name} rescaled")
    for tag, average in sorted(averages.items()):
        old, _ = row_pose(csv_rows[tag])
        print(f"  tag {tag}: x, y, z {np.round(old, 3)} -> {np.round(average.position, 3)} m")
    if dry_run:
        return

    backup_dir = parquet_path.parent / BACKUP_DIR_NAME
    backup_dir.mkdir(exist_ok=True)
    for path in (parquet_path, csv_path):
        if (backup_dir / path.name).exists():
            raise SystemExit(f"{backup_dir / path.name} already exists. Nothing changed.")
    for path in (parquet_path, csv_path):
        shutil.copy2(path, backup_dir / path.name)

    sizes = dict(recorded_sizes)
    if np.isclose(true_size, metadata["tag_size_m"]):
        sizes.pop(str(tag_id), None)
    else:
        sizes[str(tag_id)] = true_size
    metadata["tag_sizes_m"] = sizes
    metadata.setdefault("corrections", []).append(
        f"tag {tag_id} positions rescaled from a {recorded_size} m to a {true_size} m tag "
        f"by utils/fix_tag_size.py; originals in {BACKUP_DIR_NAME}/"
    )
    write_tracking_parquet(parquet_path, frames, metadata)

    first_row = next(iter(csv_rows.values()))
    names = {tag: row["name"] for tag, row in csv_rows.items()}
    write_average_csv(
        csv_path,
        first_row["bush_id"],
        first_row["cane_id"],
        float(first_row["timestamp"]),
        reference_id,
        averages,
        names,
        detection=detection,
    )


def main():
    """Fix every recording named on the command line."""
    parser = argparse.ArgumentParser(description="Correct recordings made with the wrong tag size.")
    parser.add_argument("parquet", nargs="+", help="*_frames.parquet files to fix")
    parser.add_argument("--tag", type=int, required=True, help="ID of the tag with the wrong size")
    parser.add_argument("--recorded-size", type=float, required=True, help="Size used when recording, in metres")
    parser.add_argument("--true-size", type=float, required=True, help="Real black-square size, in metres")
    parser.add_argument("--dry-run", action="store_true", help="Show the changes without writing anything")
    args = parser.parse_args()
    for path in args.parquet:
        fix_recording(path, args.tag, args.recorded_size, args.true_size, args.dry_run)


if __name__ == "__main__":
    main()
