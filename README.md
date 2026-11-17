# at-tracking

Track rigid objects with AprilTags and an Intel RealSense camera, and save their
poses to Parquet or measure them with a GUI. Forked from a Franka apple-picking setup; the code here has no
robot dependency.

## Setup

```bash
pip install -r requirements.txt
```

`reqexact.txt` lists pinned versions. The GUI also needs Tk, which on Ubuntu comes
from the system package `python3-tk`.

## GUI

```bash
python -m src.gui --config config/example.yaml
```

1. Choose a reference tag. The table shows live x, y, z of every other tag in that
   tag's frame.
2. Adjust **Exposure** until all the tags are detected, or tick **Auto**. Changes
   last until you close the window; the starting value comes from the config. The
   camera is updated 0.3 s after the slider stops moving, not during a drag.
3. Enter the **Bush ID** and **Cane ID**, and set the recording length in seconds.
4. Press **Record**, which is available while the reference tag is visible and both
   IDs are filled in.

If the camera disconnects, the GUI keeps trying to reopen it every 2 seconds, using
the same exposure. A recording in progress at the time is discarded.

Detections with an impossible pose, such as a mirror-image rotation, are ignored. If
a frame still can't be processed, it's skipped with a message in the status bar and
the camera stays connected.

If the camera stops responding without disconnecting, it needs unplugging and
plugging back in. Closing the window still ends the program in that case: if the
camera isn't released within 10 seconds, the program ends itself so the camera is
freed.

The reference tag has its own row at the top of the table. Each row's colour shows
how well that tag has been detected over the last 2 seconds, measured in the camera
frame:

| Colour | Meaning |
|---|---|
| Green | Detected in at least 70% of frames, at most 15% of them flipped, and jitter at most 10 mm |
| Yellow | Detected, but missing at least one green limit. The **Detection** column says which |
| None | Detected in under 20% of frames, or not seen for over 1 second |

Jitter is a robust standard deviation of position on the worst axis, after removing
flips. A colour only changes once the new one has lasted 1 second. **Record** needs
the reference row to be green. All of these limits are constants at the top of
[src/gui.py](src/gui.py).

**Recorded?** keeps track of which tags have been captured for the current cane, so
a cane can be covered over several recordings. After each recording, a tag gets a
checkmark if it was green over that whole recording and has a row in the CSV.
Checked rows are greyed out. If the reference wasn't green during the recording,
nothing is checked off. The status bar says which tags were checked off and why the
others weren't. Once every tag seen so far is checked, a green message says so. **New cane** clears
the checkmarks and the Cane ID, after asking for confirmation.
Checkmarks aren't saved, so they reset when the GUI closes.

The table shows a 1 second rolling median, so it's easier to read. A hidden tag keeps
its value until it has been out of view for over 1 second. Only the display is
filtered; nothing in the saved files is.

Each recording saves three files to `data/`, named
`bush<B>_cane<C>_<YYYYmmdd>_<HHMM>_*`. A second recording of the same cane in the
same minute gets a `_2` suffix, so nothing is overwritten:

| File | Contents |
|---|---|
| `*_average.csv` | One averaged pose per tag, in the reference tag's frame |
| `*_frames.parquet` | Every frame's poses in the camera frame, including the reference tag |
| `*_snapshot.png` | The first recorded frame, with tag outlines and axes |

The GUI only shows tags listed under `objects` in the config, plus `reference_tag`
and any tag in `tag_sizes_m`, so add a new tag to the config before using it. Any listed tag can be
chosen as the reference. Each tag is handled individually: the config supplies the
tag sizes, camera settings, decision margin, and display names, while multi-tag
objects and their offsets aren't applied.

CSV columns are `bush_id, cane_id, timestamp, reference_id, tag_id, name, x, y, z, qx, qy, qz, qw,
n_frames, n_rejected, pos_std_mm, detected_pct, flipped_pct, jitter_mm, well_detected`,
in metres with xyzw quaternions. Only frames showing both the tag and the reference
are averaged. Frames whose rotation is more than 15 degrees from the tag's typical
rotation are dropped first, and `n_rejected` counts them. `pos_std_mm` is the
largest per-axis standard deviation.

The last four columns describe how well the tag was detected over the recording, in
the camera frame, using the same measures as the GUI's row colours:

| Column | Meaning |
|---|---|
| `detected_pct` | Share of the recording's frames that included the tag |
| `flipped_pct` | Share of those detections that were orientation flips |
| `jitter_mm` | Robust spread of position on the worst axis, after removing flips |
| `well_detected` | `True` if the tag and the reference were both green over the recording, the same rule as the GUI's checkmark |

Filter on `well_detected` to keep only reliable rows. The Parquet metadata records
the limits used (`quality_limits`), the well-detected tag IDs, and these measures
for every tag seen, including the reference. CSVs saved before this change don't
have these columns.

### A larger reference tag

A bigger reference tag gives a steadier reference frame. Tags can have different
sizes: `tag_size_m` is the default, and `tag_sizes_m` lists the exceptions by ID,
each the edge length of the black square in metres:

```yaml
tag_size_m: 0.0254
reference_tag: 10    # optional; preselects it in the GUI
tag_sizes_m:
  10: 0.100
```

## Reconstructing a cane

A cane often takes several recordings to cover. To combine them into one position
per tag relative to the reference:

```bash
python -m src.reconstruct --bush 12 --cane 3 --folder data
```

Leave out `--folder` to choose it in a window. The script:

- Finds every `bush<B>_cane<C>_*_average.csv` in the folder.
- Uses only rows with `well_detected` set to `True`, the tags that got a checkmark.
- Combines each tag's recordings, weighted by frame count. `spread_mm` is how far the
  furthest recording is from the combined position. Over 10 mm gives a warning, since
  the cane or the reference may have moved between recordings.
- Builds one reconstruction per day, since canes bend over the season. Use
  `--date YYYYmmdd` to choose one day.
- Only combines recordings with the same reference tag: the most used one, or the
  one given with `--reference`.
- Lists tags that were recorded but never well detected.

Add `--plot` to draw the tags in 3D and open the plot in a window. Tags are joined
in ID order (0 to 1, 1 to 2, ...), with a dashed line where tags in between are
missing. The reference is a star at the origin, every axis has the same scale, and
each day gets its own colour, so bending over the season shows up. The picture is
also saved as `bush<B>_cane<C>_<dates>_reconstruction.png`. The axes are the
reference tag's: x to the tag's right, y down the tag, and z into the tag.

Each day is saved as `bush<B>_cane<C>_<YYYYmmdd>_reconstruction.csv` in the same
folder, with the reference as a row at the origin. Recordings from before
`well_detected` existed are skipped, unless you add `--include-unjudged` to use all
their tags.

## Command-line recording

Copy [config/example.yaml](config/example.yaml), then describe your tag size and
objects. Run from the repository root:

```bash
python -m src --config config/example.yaml --output data/tracking.parquet
```

| Flag | Effect |
|---|---|
| `--headless` | Run without the live window |
| `--record` | Also save the raw camera feed to an MP4 |
| `--record-output PATH` | MP4 path; defaults to `--output` with `.mp4` |

Press `q` or `Esc` in the window, or Ctrl-C, to stop and write the file.

To see how often each object was visible in a recording:

```bash
python -m src.data_collector data/tracking.parquet
```

## Output

Each row is one object in one frame, with columns `timestamp, name, x, y, z, qx,
qy, qz, qw`. Positions are in metres and quaternions are in xyzw order. A frame
has no row for an object that wasn't detected.

Poses are in the camera frame, or in the frame of `reference_tag` if the config
sets one. In reference mode nothing is saved until that tag has been seen, and
its last pose is reused if it is briefly hidden, so it should stay fixed in place.

Settings, the camera intrinsics, and the tag-to-object transforms are stored as
JSON in the Parquet footer, under the `dataset_metadata` key.

The object position is the average across its visible tags. The rotation comes
from the first visible tag in the order listed in the config, so list the most
reliable tag first.

## Layout

| File | Purpose |
|---|---|
| `src/gui.py` | GUI window |
| `src/capture.py` | Background camera loop and recording for the GUI |
| `src/measurement.py` | Relative poses, averaging, and CSV output |
| `src/__main__.py` | Command-line entry point and capture loop |
| `src/config.py` | Loads and validates the YAML config |
| `src/camera.py` | RealSense capture and MP4 recording |
| `src/detection.py` | Tag detection and tracker updates |
| `src/tracker.py` | Pose estimate for one rigid object |
| `src/data_collector.py` | Parquet writer and recording summary |
| `src/annotate.py` | Live overlay. Axis colors are X red, Y green, Z blue |
| `apriltags/something.pdf` | Printable tag36h11 sheet, IDs 0-5 |
