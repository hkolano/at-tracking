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

If the camera stops responding without disconnecting, it needs unplugging and
plugging back in. Closing the window still ends the program in that case: if the
camera isn't released within 10 seconds, the program ends itself so the camera is
freed.

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
if it's set, so add a new tag to the config before using it. Any listed tag can be
chosen as the reference. Each tag is handled individually: the config supplies the
tag size, camera settings, decision margin, and display names, while multi-tag
objects and their offsets aren't applied.

CSV columns are `bush_id, cane_id, timestamp, reference_id, tag_id, name, x, y, z, qx, qy, qz, qw,
n_frames, n_rejected, pos_std_mm`, in metres with xyzw quaternions. Only frames
showing both the tag and the reference are averaged. Frames whose rotation is more
than 15 degrees from the tag's typical rotation are dropped first, and `n_rejected`
counts them. `pos_std_mm` is the largest per-axis standard deviation.

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
