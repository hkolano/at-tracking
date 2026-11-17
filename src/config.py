"""Load and validate the YAML file describing tags and tracked objects."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.tracker import Tracker

# Tolerant enough for hand-typed values such as 0.7071.
ROTATION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: int | None = 100  # None leaves auto exposure on.


@dataclass(frozen=True)
class TrackedObjectConfig:
    name: str
    tag_ids: tuple[int, ...]
    offsets: tuple[dict, ...]


@dataclass(frozen=True)
class Config:
    tag_size_m: float
    objects: tuple[TrackedObjectConfig, ...]
    tag_family: str = "tag36h11"
    decision_margin: float = 3.0
    reference_tag: int | None = None
    camera: CameraConfig = CameraConfig()
    # tag_id -> edge length in metres, for tags that differ from tag_size_m
    # (for example a larger reference tag).
    tag_sizes_m: dict = field(default_factory=dict)

    def detection_metadata(self) -> dict:
        """Return the detection settings that every recording's metadata records."""
        return {
            "tag_family": self.tag_family,
            "tag_size_m": self.tag_size_m,
            "tag_sizes_m": {str(tag_id): size for tag_id, size in self.tag_sizes_m.items()},
            "decision_margin_threshold": self.decision_margin,
        }

    def build_trackers(self) -> list[Tracker]:
        """Return a fresh Tracker for each object."""
        return [Tracker(obj.name, obj.tag_ids, obj.offsets) for obj in self.objects]


def _parse_tag(raw: dict, object_name: str) -> tuple[int, dict]:
    if "id" not in raw:
        raise ValueError(f"Object '{object_name}' has a tag with no id")
    tag_id = int(raw["id"])

    pos = np.asarray(raw.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot = np.asarray(raw.get("rot", np.eye(3)), dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError(f"Tag {tag_id}: pos must have 3 values")
    if rot.shape != (3, 3):
        raise ValueError(f"Tag {tag_id}: rot must be a 3x3 matrix")

    is_orthonormal = np.allclose(rot @ rot.T, np.eye(3), atol=ROTATION_TOLERANCE)
    is_right_handed = abs(np.linalg.det(rot) - 1.0) < ROTATION_TOLERANCE
    if not (is_orthonormal and is_right_handed):
        raise ValueError(f"Tag {tag_id}: rot is not a valid rotation matrix")

    return tag_id, {"pos": pos, "rot": rot}


def _parse_object(raw: dict) -> TrackedObjectConfig:
    name = raw.get("name")
    if not name:
        raise ValueError("Every object needs a name")
    tags = raw.get("tags")
    if not tags:
        raise ValueError(f"Object '{name}' needs at least one tag")

    parsed = [_parse_tag(tag, name) for tag in tags]
    return TrackedObjectConfig(
        name=str(name),
        tag_ids=tuple(tag_id for tag_id, _ in parsed),
        offsets=tuple(offset for _, offset in parsed),
    )


def _parse_tag_sizes(raw) -> dict:
    """Return tag_id -> size in metres from the optional tag_sizes_m mapping."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("tag_sizes_m must map tag IDs to sizes, for example {10: 0.1}")
    sizes = {}
    for tag_id, size in raw.items():
        if size is None or float(size) <= 0:
            raise ValueError(f"tag_sizes_m: tag {tag_id} needs a positive size")
        sizes[int(tag_id)] = float(size)
    return sizes


def load_config(path: str | Path) -> Config:
    """Read a YAML config file and check it for mistakes."""
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    tag_size_m = raw.get("tag_size_m")
    if tag_size_m is None or float(tag_size_m) <= 0:
        raise ValueError("tag_size_m is required and must be positive")

    objects = tuple(_parse_object(obj) for obj in raw.get("objects") or [])
    if not objects:
        raise ValueError("At least one object is required")

    names = [obj.name for obj in objects]
    if len(set(names)) != len(names):
        raise ValueError("Object names must be unique")

    # A tag on two objects would silently tie those objects' poses together.
    tag_ids = [tag_id for obj in objects for tag_id in obj.tag_ids]
    duplicates = sorted({tag_id for tag_id in tag_ids if tag_ids.count(tag_id) > 1})
    if duplicates:
        raise ValueError(f"Tag IDs used more than once: {duplicates}")

    reference_tag = raw.get("reference_tag")
    if reference_tag is not None:
        reference_tag = int(reference_tag)
        if reference_tag in tag_ids:
            raise ValueError(f"Reference tag {reference_tag} is also assigned to an object")

    return Config(
        tag_size_m=float(tag_size_m),
        objects=objects,
        tag_family=str(raw.get("tag_family", "tag36h11")),
        decision_margin=float(raw.get("decision_margin", 3.0)),
        reference_tag=reference_tag,
        camera=CameraConfig(**(raw.get("camera") or {})),
        tag_sizes_m=_parse_tag_sizes(raw.get("tag_sizes_m")),
    )
