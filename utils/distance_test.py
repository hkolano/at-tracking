"""Report the measured distance between tracked tags, to compare with a ruler.

Run from the repository root:
    python -m utils.distance_test --config config/example.yaml
"""

import argparse
import itertools
import time

import cv2
import numpy as np

from src.camera import RealSenseCamera
from src.config import load_config
from src.detection import TagPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--save-frame", default=None, help="Save the last camera frame to this image path")
    args = parser.parse_args()

    config = load_config(args.config)
    trackers = config.build_trackers()
    camera = RealSenseCamera(config.camera)
    pipeline = TagPipeline(
        trackers,
        camera.camera_params,
        config.tag_size_m,
        tag_family=config.tag_family,
        decision_margin=config.decision_margin,
        reference_id=config.reference_tag,
    )

    positions = {tracker.name: [] for tracker in trackers}
    pair_distances = {}
    frames = 0
    frame = None
    try:
        end_time = time.monotonic() + args.seconds
        while time.monotonic() < end_time:
            frame = camera.read()
            if frame is None:
                continue
            frames += 1
            pipeline.process_frame(frame)

            visible = [tracker for tracker in trackers if tracker.pose is not None]
            for tracker in visible:
                positions[tracker.name].append(np.asarray(tracker.pose["pos"], dtype=np.float64))
            for first, second in itertools.combinations(visible, 2):
                distance = np.linalg.norm(first.pose["pos"] - second.pose["pos"])
                pair_distances.setdefault((first.name, second.name), []).append(distance)
    finally:
        camera.close()

    if args.save_frame and frame is not None:
        cv2.imwrite(args.save_frame, frame)

    print(f"{frames} frames, tag size {config.tag_size_m * 1000:.1f} mm")
    print("\nTags seen:")
    for name, samples in positions.items():
        if not samples:
            continue
        samples = np.asarray(samples)
        x, y, z = samples.mean(axis=0)
        jitter_mm = samples.std(axis=0).max() * 1000
        print(f"  {name}: seen in {len(samples)}/{frames} frames, "
              f"position [{x:+.4f}, {y:+.4f}, {z:+.4f}] m, jitter {jitter_mm:.2f} mm")

    print("\nDistances between tags:")
    for (first, second), distances in pair_distances.items():
        print(f"  {first} to {second}: {np.mean(distances) * 1000:.1f} mm "
              f"(jitter {np.std(distances) * 1000:.2f} mm, {len(distances)} frames)")
    if not pair_distances:
        print("  None: need two tags visible in the same frame")


if __name__ == "__main__":
    main()
