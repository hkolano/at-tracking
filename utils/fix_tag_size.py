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
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from src.measurement import (
    QualityLimits,
    TagQuality,
    average_poses,
    judge_recording,
    quality_columns,
    write_average_csv,
)

BACKUP_DIR_NAME = "before_tag_size_fix"
POSITION_TOLERANCE_M = 1e-5  # CSVs store 6 decimals, so 1e-6 m rounding.
QUATERNION_TOLERANCE = 1e-4


def read_recording(parquet_path):
    """Return a recording's frames as a DataFrame, and its metadata dict."""
    table = pq.read_table(parquet_path)
    metadata = json.loads(table.schema.metadata[b"dataset_metadata"])
    return table.to_pandas(), metadata


def write_recording(parquet_path, frames, metadata):
    """Write frames and metadata to Parquet in the same layout as DataCollector."""
    table = pa.Table.from_pandas(frames, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"dataset_metadata"] = json.dumps(
        metadata, sort_keys=True, default=str
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(schema_metadata), parquet_path)


def average_relative_poses(frames, reference_name, tag_ids):
    """Rebuild the CSV averages: each tag's pose relative to the reference, per frame.

    Args:
        frames:         Camera-frame rows, one per tag per frame.
        reference_name: Name of the reference tag's rows.
        tag_ids:        dict of name -> tag ID.

    Returns:
        dict of tag_id -> AveragedPose, as the GUI computes it.
    """
    positions, rotations = {}, {}
    for _, frame in frames.groupby("timestamp", sort=True):
        reference_rows = frame[frame["name"] == reference_name]
        if reference_rows.empty:
            continue
        reference = reference_rows.iloc[0]
        reference_rotation = R.from_quat(reference[["qx", "qy", "qz", "qw"]].to_numpy(float)).as_matrix()
        reference_position = reference[["x", "y", "z"]].to_numpy(float)
        for _, row in frame[frame["name"] != reference_name].iterrows():
            rotation = R.from_quat(row[["qx", "qy", "qz", "qw"]].to_numpy(float)).as_matrix()
            position = row[["x", "y", "z"]].to_numpy(float)
            tag_id = tag_ids[row["name"]]
            positions.setdefault(tag_id, []).append(reference_rotation.T @ (position - reference_position))
            rotations.setdefault(tag_id, []).append(reference_rotation.T @ rotation)
    return {tag_id: average_poses(positions[tag_id], rotations[tag_id]) for tag_id in positions}


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
        position = np.array([float(row[axis]) for axis in "xyz"])
        quaternion = np.array([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
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
    csv_path = parquet_path.with_name(parquet_path.name.replace("_frames.parquet", "_average.csv"))
    frames, metadata = read_recording(parquet_path)
    csv_rows = read_csv_rows(csv_path)

    recorded_sizes = metadata.get("tag_sizes_m") or {}
    if str(tag_id) in recorded_sizes:
        size_in_file = recorded_sizes[str(tag_id)]
    else:
        size_in_file = metadata["tag_size_m"]
    if not np.isclose(size_in_file, recorded_size):
        raise SystemExit(
            f"{parquet_path}: tag {tag_id} was recorded at {size_in_file} m, not "
            f"{recorded_size} m. Nothing changed. Already fixed?"
        )

    tag_ids = {name: int(key) for key, name in metadata["tag_names"].items()}
    tag_name = metadata["tag_names"].get(str(tag_id), f"tag_{tag_id}")
    tag_ids.setdefault(tag_name, tag_id)
    reference_id = metadata["reference_tag_id"]
    reference_name = metadata["tag_names"].get(str(reference_id), f"tag_{reference_id}")

    # Check the averaging is rebuilt exactly as the GUI did it before changing anything.
    if not matches_csv(average_relative_poses(frames, reference_name, tag_ids), csv_rows):
        raise SystemExit(f"{csv_path}: couldn't reproduce the existing CSV. Nothing changed.")

    is_tag = frames["name"] == tag_name
    frames.loc[is_tag, ["x", "y", "z"]] *= true_size / recorded_size
    averages = average_relative_poses(frames, reference_name, tag_ids)
    detection = rescale_detection(
        metadata, tag_id, true_size / recorded_size, reference_id, set(averages)
    )

    print(f"{parquet_path.name}: {int(is_tag.sum())} rows of {tag_name} rescaled")
    for tag, average in sorted(averages.items()):
        old = [float(csv_rows[tag][axis]) for axis in "xyz"]
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
    write_recording(parquet_path, frames, metadata)

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
