"""Record tracked object poses: python -m at_tracking --config config/example.yaml"""

import argparse
import signal
import time
from pathlib import Path
from threading import Event

import cv2
from scipy.spatial.transform import Rotation as R

from src import annotate
from src.camera import RealSenseCamera, VideoRecorder
from src.config import load_config
from src.data_collector import DataCollector
from src.detection import TagPipeline

ESCAPE_KEY = 27


def _parse_args():
    parser = argparse.ArgumentParser(description="Record AprilTag object poses with Unix timestamps.")
    parser.add_argument("--config", required=True, help="YAML file describing tags and objects")
    parser.add_argument("--output", default="tracking.parquet", help="Output Parquet path")
    parser.add_argument("--headless", action="store_true", help="Run without the live window")
    parser.add_argument("--record", action="store_true", help="Also save the raw camera feed to an MP4")
    parser.add_argument("--record-output", default=None, help="MP4 path; defaults to --output with .mp4")
    return parser.parse_args()


def _build_metadata(config, camera, trackers):
    return {
        "capture_start_timestamp": time.time(),
        "coordinate_frame": "reference_tag" if config.reference_tag is not None else "camera",
        "reference_tag_id": config.reference_tag,
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "tag_family": config.tag_family,
        "tag_size_m": config.tag_size_m,
        "decision_margin_threshold": config.decision_margin,
        "camera": {
            "stream": "color",
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps,
            "manual_exposure": config.camera.exposure,
            "intrinsics_matrix": camera.K.tolist(),
            "distortion_coefficients": camera.dist_coeffs.tolist(),
        },
        "tracker_tag_to_object_transforms": {
            tracker.name: {
                str(tag_id): transform.tolist()
                for tag_id, transform in tracker.offsets.items()
            }
            for tracker in trackers
        },
    }


def main():
    args = _parse_args()
    config = load_config(args.config)

    stop_requested = Event()
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

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
    collector = DataCollector(metadata=_build_metadata(config, camera, trackers))

    recorder = None
    if args.record:
        video_path = Path(args.record_output) if args.record_output else Path(args.output).with_suffix(".mp4")
        recorder = VideoRecorder(video_path, config.camera.fps)
        print(f"Recording camera feed to {video_path}")

    try:
        while not stop_requested.is_set():
            frame = camera.read()
            if frame is None:
                continue
            timestamp = time.time()

            if recorder is not None:
                recorder.write(frame)  # before annotation, so the video stays clean

            tag_dict = pipeline.process_frame(frame)

            for tracker in trackers:
                if tracker.pose is None:
                    continue
                x, y, z = tracker.pose["pos"]
                quat = R.from_matrix(tracker.pose["rot"]).as_quat()
                collector.update(timestamp, tracker.name, x, y, z, *quat)

            if not args.headless:
                annotate.annotate_frame(
                    frame,
                    tag_dict,
                    last_reference_pose=pipeline.last_reference_pose,
                    trackers=trackers,
                    camera_matrix=camera.K,
                    dist_coeffs=camera.dist_coeffs,
                    use_reference_frame=pipeline.use_reference_frame,
                )
                cv2.imshow("AprilTag Tracker", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ESCAPE_KEY):
                    stop_requested.set()
    finally:
        collector.dump(
            args.output,
            metadata={"capture_end_timestamp": time.time(), "row_count": len(collector.rows)},
        )
        print(f"Wrote tracking data to {args.output}")
        if recorder is not None:
            recorder.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
