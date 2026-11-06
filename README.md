# at-tracking

Track rigid objects with AprilTags and an Intel RealSense camera, and save their
poses to Parquet. Forked from a Franka apple-picking setup; the code here has no
robot dependency.

## Setup

```bash
pip install -r requirements.txt
```

`reqexact.txt` lists pinned versions.

## Usage

Copy [config/example.yaml](config/example.yaml), then describe your tag size and
objects. Run from the repository root:

```bash
python -m at_tracking --config config/example.yaml --output tracking.parquet
```

| Flag | Effect |
|---|---|
| `--headless` | Run without the live window |
| `--record` | Also save the raw camera feed to an MP4 |
| `--record-output PATH` | MP4 path; defaults to `--output` with `.mp4` |

Press `q` or `Esc` in the window, or Ctrl-C, to stop and write the file.

To see how often each object was visible in a recording:

```bash
python -m at_tracking.data_collector tracking.parquet
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
| `at_tracking/__main__.py` | Command-line entry point and capture loop |
| `at_tracking/config.py` | Loads and validates the YAML config |
| `at_tracking/camera.py` | RealSense capture and MP4 recording |
| `at_tracking/detection.py` | Tag detection and tracker updates |
| `at_tracking/tracker.py` | Pose estimate for one rigid object |
| `at_tracking/data_collector.py` | Parquet writer and recording summary |
| `at_tracking/annotate.py` | Live overlay. Axis colors are X red, Y green, Z blue |
| `apriltags/something.pdf` | Printable tag36h11 sheet, IDs 0-5 |
