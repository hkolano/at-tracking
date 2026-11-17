"""Combine a cane's recordings into one set of tag positions relative to the reference.

A cane usually takes several recordings to cover, each capturing some tags well.
This script finds every recording of one bush and cane in a folder, keeps only the
tags marked well_detected (the GUI's checkmarks), and combines them into one
position per tag, in the reference tag's frame.

Canes bend over the season, so recordings are only combined within a day. Each day
gets its own reconstruction, saved next to the recordings as
bush<B>_cane<C>_<YYYYmmdd>_reconstruction.csv.

Run from the repository root:
    python -m src.reconstruct --bush 12 --cane 3 --folder data --plot
Leave out --folder to choose it in a window. --plot also draws the tags in 3D,
joined in tag ID order, saves the picture, and opens it in a window.
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.measurement import AVERAGE_SUFFIX, recording_stem, row_pose

# Warn when a tag's recordings disagree by more than this, which suggests the cane
# or the reference moved between recordings.
SPREAD_WARNING_MM = 10.0

RECONSTRUCTION_COLUMNS = [
    "bush_id", "cane_id", "date", "reference_id", "tag_id", "name",
    "x", "y", "z", "qx", "qy", "qz", "qw",
    "n_recordings", "n_frames", "spread_mm", "recordings",
]


@dataclass
class Measurement:
    """One tag's averaged pose from one recording."""

    recording: str          # CSV file name
    reference_id: int
    tag_id: int
    name: str
    position: np.ndarray    # (3,) metres, reference frame
    quaternion: np.ndarray  # (4,) xyzw
    n_frames: int


@dataclass
class TagEstimate:
    """One tag's combined position over a day's recordings."""

    tag_id: int
    name: str
    position: np.ndarray
    quaternion: np.ndarray
    n_frames: int
    spread_mm: float        # largest distance of one recording from the combined position
    recordings: list


@dataclass
class Reconstruction:
    """All tags of one cane on one day, relative to one reference tag."""

    date: str               # YYYYmmdd
    reference_id: int
    tags: dict = field(default_factory=dict)          # tag_id -> TagEstimate
    recordings: list = field(default_factory=list)    # CSV names that contributed
    not_well_detected: dict = field(default_factory=dict)  # tag_id -> name, never usable
    warnings: list = field(default_factory=list)


def find_recordings(folder, bush_id, cane_id):
    """Return the average CSVs for one bush and cane in a folder, oldest first."""
    return sorted(Path(folder).glob(f"{recording_stem(bush_id, cane_id)}_*{AVERAGE_SUFFIX}"))


def read_recording(path, include_unjudged=False):
    """Read one average CSV.

    Returns:
        (date as YYYYmmdd, usable Measurements, tag_id -> name for rows that were
         not well detected, warning text or "")
    """
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None, [], {}, f"{path.name}: no tags were recorded."

    judged = "well_detected" in rows[0]
    warning = ""
    if not judged:
        if not include_unjudged:
            return None, [], {}, (
                f"{path.name}: recorded before tags were judged, so skipped. "
                "Use --include-unjudged to use all of its tags."
            )
        warning = f"{path.name}: recorded before tags were judged, so all its tags were used."

    date = datetime.fromtimestamp(float(rows[0]["timestamp"])).strftime("%Y%m%d")
    usable, rejected = [], {}
    for row in rows:
        tag_id = int(row["tag_id"])
        if judged and row["well_detected"] != "True":
            rejected[tag_id] = row["name"]
            continue
        position, quaternion = row_pose(row)
        usable.append(Measurement(
            recording=path.name,
            reference_id=int(row["reference_id"]),
            tag_id=tag_id,
            name=row["name"],
            position=position,
            quaternion=quaternion,
            n_frames=int(row["n_frames"]),
        ))
    return date, usable, rejected, warning


def combine(measurements):
    """Combine one tag's measurements, weighting each recording by its frame count."""
    weights = np.array([measurement.n_frames for measurement in measurements], dtype=float)
    positions = np.array([measurement.position for measurement in measurements])
    position = np.average(positions, axis=0, weights=weights)
    rotations = R.from_quat([measurement.quaternion for measurement in measurements])
    spread = np.linalg.norm(positions - position, axis=1).max() * 1000.0
    first = measurements[0]
    return TagEstimate(
        tag_id=first.tag_id,
        name=first.name,
        position=position,
        quaternion=rotations.mean(weights=weights).as_quat(),
        n_frames=int(weights.sum()),
        spread_mm=float(spread),
        recordings=[measurement.recording for measurement in measurements],
    )


def reconstruct(paths, include_unjudged=False, reference_id=None):
    """Build one Reconstruction per day from a cane's recordings.

    Positions from different recordings can only be combined if they share a
    reference tag. Each day uses reference_id if given, or else the reference used
    by most of that day's well-detected measurements; recordings with another
    reference are left out with a warning.

    Returns:
        (list of Reconstructions, oldest first, list of general warnings)
    """
    by_date, rejected_by_date, general_warnings = {}, {}, []
    for path in paths:
        date, usable, rejected, warning = read_recording(path, include_unjudged)
        if warning:
            general_warnings.append(warning)
        if date is None:
            continue
        by_date.setdefault(date, []).extend(usable)
        rejected_by_date.setdefault(date, {}).update(rejected)

    reconstructions = []
    for date in sorted(by_date):
        measurements = by_date[date]
        references = [measurement.reference_id for measurement in measurements]
        if reference_id is not None:
            chosen = reference_id
        elif references:
            chosen = max(set(references), key=references.count)
        else:
            chosen = None
        result = Reconstruction(date=date, reference_id=chosen)

        others = sorted({m.recording for m in measurements if m.reference_id != chosen})
        if others:
            result.warnings.append(
                f"Left out, as they used a different reference tag: {', '.join(others)}"
            )
        measurements = [m for m in measurements if m.reference_id == chosen]

        by_tag = {}
        for measurement in measurements:
            by_tag.setdefault(measurement.tag_id, []).append(measurement)
        for tag_id, tag_measurements in sorted(by_tag.items()):
            estimate = combine(tag_measurements)
            result.tags[tag_id] = estimate
            if estimate.spread_mm > SPREAD_WARNING_MM:
                result.warnings.append(
                    f"{estimate.name}: recordings disagree by up to {estimate.spread_mm:.0f} mm. "
                    "The cane or the reference may have moved between them."
                )
        result.recordings = sorted({m.recording for m in measurements})
        result.not_well_detected = {
            tag_id: name
            for tag_id, name in rejected_by_date[date].items()
            if tag_id not in result.tags
        }
        reconstructions.append(result)
    return reconstructions, general_warnings


def write_reconstruction_csv(path, bush_id, cane_id, result):
    """Write one row per tag, plus the reference at the origin, in the reference frame."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RECONSTRUCTION_COLUMNS)
        writer.writerow([
            bush_id, cane_id, result.date, result.reference_id, result.reference_id,
            "reference", *["0.000000"] * 6, "1.000000",
            len(result.recordings), "", "", ";".join(result.recordings),
        ])
        for estimate in result.tags.values():
            writer.writerow([
                bush_id,
                cane_id,
                result.date,
                result.reference_id,
                estimate.tag_id,
                estimate.name,
                *(f"{value:.6f}" for value in estimate.position),
                *(f"{value:.6f}" for value in estimate.quaternion),
                len(estimate.recordings),
                estimate.n_frames,
                f"{estimate.spread_mm:.1f}",
                ";".join(estimate.recordings),
            ])


def print_reconstruction(result):
    """Print one day's reconstruction as a table."""
    print(f"\n{result.date}: {len(result.recordings)} recordings, "
          f"relative to reference tag {result.reference_id}")
    print(f"  {'tag':>4}  {'name':<12} {'x (m)':>8} {'y (m)':>8} {'z (m)':>8}  "
          f"{'recordings':>10}  {'spread':>8}")
    for estimate in result.tags.values():
        x, y, z = estimate.position
        print(f"  {estimate.tag_id:>4}  {estimate.name:<12} {x:+8.3f} {y:+8.3f} {z:+8.3f}  "
              f"{len(estimate.recordings):>10}  {estimate.spread_mm:>5.1f} mm")
    if not result.tags:
        print("  No well-detected tags.")
    if result.not_well_detected:
        names = ", ".join(f"{name} ({tag_id})" for tag_id, name in sorted(result.not_well_detected.items()))
        print(f"  Never well detected, so missing: {names}")
    for warning in result.warnings:
        print(f"  Warning: {warning}")


def plot_reconstructions(reconstructions, title, output_path=None, show=False):
    """Plot each day's tags in 3D, joined in tag ID order, all days on one set of axes.

    The reference tag is drawn as a star at the origin. Tags are joined in ID order
    (0 to 1, 1 to 2, and so on). A dashed line means tags are missing between the
    two, because they were never well detected that day. All axes use the same scale,
    so the cane's shape isn't distorted.

    Args:
        reconstructions: Reconstructions to draw, one colour per day.
        title:           Figure title.
        output_path:     Save the figure here, if given.
        show:            Open the figure in a window, and wait until it is closed.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # Draw without a window, e.g. on a machine with no display.
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 8))
    axes = figure.add_subplot(projection="3d")
    colours = plt.get_cmap("tab10")
    all_points = [np.zeros(3)]

    for index, result in enumerate(reconstructions):
        colour = colours(index % 10)
        points = {result.reference_id: np.zeros(3)}
        points.update({tag_id: estimate.position for tag_id, estimate in result.tags.items()})
        ordered = sorted(points)
        all_points.extend(points.values())

        for first, second in zip(ordered, ordered[1:]):
            segment = np.array([points[first], points[second]])
            style = "-" if second == first + 1 else "--"
            axes.plot(*segment.T, style, color=colour, linewidth=2)

        tags = [tag_id for tag_id in ordered if tag_id != result.reference_id]
        if tags:
            positions = np.array([points[tag_id] for tag_id in tags])
            axes.scatter(*positions.T, color=colour, s=40, depthshade=False,
                         label=f"{result.date} ({len(result.recordings)} recordings)")
        axes.scatter(0, 0, 0, color=colour, marker="*", s=250, depthshade=False)
        for tag_id in ordered:
            axes.text(*points[tag_id], f"  {tag_id}", color=colour, fontsize=10)

    # Equal scale on every axis: a cube centred on the data.
    all_points = np.array(all_points)
    centre = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    half = max(np.ptp(all_points, axis=0).max() / 2, 0.05) * 1.1
    axes.set_xlim(centre[0] - half, centre[0] + half)
    axes.set_ylim(centre[1] - half, centre[1] + half)
    axes.set_zlim(centre[2] - half, centre[2] + half)
    axes.set_box_aspect((1, 1, 1))

    reference_ids = sorted({result.reference_id for result in reconstructions})
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")
    axes.set_title(f"{title}\nrelative to reference tag {', '.join(map(str, reference_ids))} (star)")
    if any(result.tags for result in reconstructions):
        axes.legend(loc="upper left")

    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def choose_folder():
    """Ask for a folder in a window; return its path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Folder with recordings", initialdir="data")
    root.destroy()
    return folder or None


def main():
    """Reconstruct a cane from the recordings in a folder."""
    parser = argparse.ArgumentParser(description="Combine a cane's recordings into one set of tag positions.")
    parser.add_argument("--bush", required=True, help="Bush ID, as typed in the GUI")
    parser.add_argument("--cane", required=True, help="Cane ID, as typed in the GUI")
    parser.add_argument("--folder", help="Folder with the recordings; leave out to choose in a window")
    parser.add_argument("--date", help="Only this day, as YYYYmmdd")
    parser.add_argument("--reference", type=int, help="Reference tag ID; defaults to the most used one")
    parser.add_argument(
        "--include-unjudged",
        action="store_true",
        help="Use every tag in recordings made before tags were judged well detected",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Draw the tags in 3D, save the picture next to the recordings, and open it",
    )
    args = parser.parse_args()

    folder = args.folder or choose_folder()
    if folder is None:
        sys.exit("No folder chosen.")
    paths = find_recordings(folder, args.bush, args.cane)
    if not paths:
        sys.exit(f"No recordings of bush {args.bush}, cane {args.cane} in {folder}.")

    reconstructions, warnings = reconstruct(paths, args.include_unjudged, args.reference)
    if args.date:
        reconstructions = [result for result in reconstructions if result.date == args.date]
    for warning in warnings:
        print(f"Warning: {warning}")
    if not reconstructions:
        print("Nothing to reconstruct.")
        sys.exit(1)

    for result in reconstructions:
        print_reconstruction(result)
        output = Path(folder) / (
            f"{recording_stem(args.bush, args.cane)}_{result.date}_reconstruction.csv"
        )
        write_reconstruction_csv(output, args.bush, args.cane, result)
        print(f"  Saved {output}")

    if args.plot:
        dates = [result.date for result in reconstructions]
        span = dates[0] if len(dates) == 1 else f"{dates[0]}-{dates[-1]}"
        stem = f"{recording_stem(args.bush, args.cane)}_{span}"
        picture = Path(folder) / f"{stem}_reconstruction.png"
        title = f"Bush {args.bush}, cane {args.cane}"
        print(f"\nSaved {picture}. Close the plot window to finish.")
        plot_reconstructions(reconstructions, title, output_path=picture, show=True)


if __name__ == "__main__":
    main()
