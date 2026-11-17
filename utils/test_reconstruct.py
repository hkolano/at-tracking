"""Tests for combining a cane's recordings.

Run from the repository root:
    python -m pytest utils/test_reconstruct.py
"""

import csv
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from src.measurement import AveragedPose, TagQuality, quality_columns, write_average_csv
from src.reconstruct import find_recordings, reconstruct

DAY_1 = datetime(2026, 6, 1, 10, 0)
DAY_2 = datetime(2026, 7, 1, 10, 0)
GOOD = TagQuality(0.9, 0.0, 3.0, 0.0)


def save(folder, started, tags, bush="12", cane="3", reference=0, judged=True):
    """Write an average CSV like the GUI does.

    Args:
        tags: dict of tag_id -> (position, n_frames, well_detected).
    """
    averages = {
        tag_id: AveragedPose(np.array(position), np.array([0, 0, 0, 1.0]), frames, 0, 1.0)
        for tag_id, (position, frames, _) in tags.items()
    }
    detection = None
    if judged:
        detection = {tag_id: quality_columns(GOOD, well) for tag_id, (_, _, well) in tags.items()}
    path = folder / f"bush{bush}_cane{cane}_{started:%Y%m%d_%H%M}_average.csv"
    names = {tag_id: f"cane_{tag_id}" for tag_id in tags}
    write_average_csv(path, bush, cane, started.timestamp(), reference, averages, names, detection)
    if not judged:  # Remove the quality columns, like a CSV from before they existed.
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        with open(path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(row[:16] for row in rows)
    return path


def test_finds_only_this_bush_and_cane(tmp_path):
    """Other bushes, other canes and similar IDs are ignored."""
    wanted = save(tmp_path, DAY_1, {1: ([0, 0, 1], 10, True)})
    save(tmp_path, DAY_1.replace(minute=1), {1: ([0, 0, 1], 10, True)}, cane="31")
    save(tmp_path, DAY_1.replace(minute=2), {1: ([0, 0, 1], 10, True)}, bush="1")
    assert find_recordings(tmp_path, "12", "3") == [wanted]


def test_combines_well_detected_tags_across_recordings(tmp_path):
    """Each tag comes from the recordings where it was well detected, weighted by frames."""
    save(tmp_path, DAY_1, {1: ([0.10, 0, 0], 30, True), 2: ([0.5, 0, 0], 30, False)})
    save(tmp_path, DAY_1.replace(minute=5), {1: ([0.13, 0, 0], 10, True), 2: ([0.20, 0, 0], 20, True)})
    save(tmp_path, DAY_1.replace(minute=9), {3: ([0.30, 0, 0], 20, False)})

    (result,), warnings = reconstruct(find_recordings(tmp_path, "12", "3"))

    assert warnings == []
    assert result.date == "20260601" and result.reference_id == 0
    assert set(result.tags) == {1, 2}
    assert result.tags[1].position == pytest.approx([0.1075, 0, 0])  # (0.10*30 + 0.13*10) / 40
    assert result.tags[1].spread_mm == pytest.approx(22.5)
    assert result.tags[2].position == pytest.approx([0.20, 0, 0])  # the poor 0.5 is ignored
    assert result.not_well_detected == {3: "cane_3"}
    assert any("cane_1: recordings disagree" in warning for warning in result.warnings)


def test_each_day_is_its_own_reconstruction(tmp_path):
    """A cane that bent between visits isn't averaged across them."""
    save(tmp_path, DAY_1, {1: ([0.10, 0, 0], 10, True)})
    save(tmp_path, DAY_2, {1: ([0.30, 0, 0], 10, True)})
    results, _ = reconstruct(find_recordings(tmp_path, "12", "3"))
    assert [result.date for result in results] == ["20260601", "20260701"]
    assert [result.tags[1].position[0] for result in results] == pytest.approx([0.10, 0.30])


def test_recordings_with_another_reference_are_left_out(tmp_path):
    """Positions relative to different references can't be combined."""
    save(tmp_path, DAY_1, {1: ([0.1, 0, 0], 10, True), 2: ([0.2, 0, 0], 10, True)})
    other = save(tmp_path, DAY_1.replace(minute=5), {1: ([0.9, 0, 0], 10, True)}, reference=7)
    (result,), _ = reconstruct(find_recordings(tmp_path, "12", "3"))
    assert result.reference_id == 0
    assert result.tags[1].position[0] == pytest.approx(0.1)
    assert any(other.name in warning for warning in result.warnings)


def test_unjudged_recordings_are_skipped_unless_asked(tmp_path):
    """Older CSVs without well_detected are only used on request."""
    save(tmp_path, DAY_1, {1: ([0.1, 0, 0], 10, False)}, judged=False)
    paths = find_recordings(tmp_path, "12", "3")

    results, warnings = reconstruct(paths)
    assert results == [] and "--include-unjudged" in warnings[0]

    (result,), warnings = reconstruct(paths, include_unjudged=True)
    assert set(result.tags) == {1}
    assert "all its tags were used" in warnings[0]


def test_command_line_writes_a_reconstruction(tmp_path):
    """The script prints the table and saves a CSV with the reference at the origin."""
    save(tmp_path, DAY_1, {1: ([0.1, 0.2, 0.3], 10, True)})
    output = subprocess.run(
        [sys.executable, "-m", "src.reconstruct", "--bush", "12", "--cane", "3",
         "--folder", str(tmp_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "cane_1" in output and "+0.100" in output

    with open(tmp_path / "bush12_cane3_20260601_reconstruction.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["name"] for row in rows] == ["reference", "cane_1"]
    assert rows[0]["x"] == "0.000000"
    assert rows[1]["n_recordings"] == "1"
    assert [float(rows[1][axis]) for axis in "xyz"] == pytest.approx([0.1, 0.2, 0.3])


def test_plot_saves_a_picture_with_gaps_and_several_days(tmp_path):
    """Days are overlaid; a missing tag is bridged rather than breaking the plot."""
    from src.reconstruct import plot_reconstructions

    save(tmp_path, DAY_1, {1: ([0.1, 0, 0], 10, True), 3: ([0.3, 0.1, 0], 10, True)})
    save(tmp_path, DAY_2, {1: ([0.1, 0, 0.05], 10, True), 2: ([0.2, 0.05, 0.1], 10, True)})
    results, _ = reconstruct(find_recordings(tmp_path, "12", "3"))
    picture = tmp_path / "plot.png"

    plot_reconstructions(results, "Bush 12, cane 3", output_path=picture)

    assert picture.stat().st_size > 10_000
