"""OpenCV overlays for AprilTag detection and tracked object poses."""

import cv2
import numpy as np

# BGR colors used for on-screen overlays.
COLOR_TAG_OUTLINE = (0, 255, 0)
COLOR_AXIS_X = (0, 0, 255)
COLOR_AXIS_Y = (0, 255, 0)
COLOR_AXIS_Z = (255, 0, 0)
COLOR_TRACKER_ORIGIN = (255, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_WAITING = (0, 0, 255)
COLOR_MODE = (0, 255, 255)
COLOR_TAG_NAME = (0, 255, 255)


def reference_extrinsics(reference_pose):
    """Return OpenCV rvec/tvec for projecting reference-frame points into the image."""
    rotation_cam_ref = np.asarray(reference_pose.pose_R, dtype=np.float64)
    translation_cam_ref = np.asarray(reference_pose.pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation_cam_ref)
    return rvec, translation_cam_ref


def draw_tag_outlines(frame, tag_dict):
    """Outline each detected tag and label it with its ID."""
    for tag_id, tag in tag_dict.items():
        # int32 avoids cv2.polylines assertion failures on 64-bit builds.
        corners = tag.corners.astype(np.int32)
        cv2.polylines(frame, [corners], True, COLOR_TAG_OUTLINE, 2)

        label_pos = tuple(corners.ravel()[:2])
        cv2.putText(
            frame,
            f"ID: {tag_id}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TAG_OUTLINE,
            2,
        )


def draw_reference_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.03):
    """Draw the reference tag coordinate frame and label its origin."""
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 3)

    origin_3d = np.zeros((1, 3), dtype=np.float64)
    origin_2d, _ = cv2.projectPoints(
        origin_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    ox, oy = int(origin_2d[0, 0, 0]), int(origin_2d[0, 0, 1])
    cv2.putText(
        frame,
        "Ref Origin",
        (ox + 10, oy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_TEXT,
        2,
    )


def draw_tag_axes(
    frame, tag_dict, camera_matrix, dist_coeffs, reference_id=None, names=None, axis_length=0.03
):
    """Draw outlines, names, and a coordinate frame on every detected tag."""
    names = names or {}
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    draw_tag_outlines(frame, tag_dict)
    for tag_id, tag in tag_dict.items():
        rvec, tvec = reference_extrinsics(tag)
        if tag_id == reference_id:
            draw_reference_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
            continue

        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 2)
        center_x, center_y = tag.center.astype(np.int32)
        cv2.putText(
            frame,
            names.get(tag_id, f"tag_{tag_id}"),
            (int(center_x) + 10, int(center_y) + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TAG_NAME,
            2,
        )


def draw_tracker_overlay(
    frame, tracker, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.06
):
    """Project a tracker's full local coordinate frame into the image."""
    position = np.asarray(tracker.pose["pos"], dtype=np.float64).reshape(1, 3)
    rotation = np.asarray(tracker.pose["rot"], dtype=np.float64)

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    # Object-frame origin and unit-axis endpoints, transformed into reference frame.
    axis_points_obj = np.array(
        [
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # +X
            [0, axis_length, 0],  # +Y
            [0, 0, axis_length],  # +Z
        ],
        dtype=np.float64,
    )
    axis_points_ref = (rotation @ axis_points_obj.T).T + position

    image_points, _ = cv2.projectPoints(
        axis_points_ref, rvec, tvec, camera_matrix, dist_coeffs
    )
    ox, oy = int(image_points[0, 0, 0]), int(image_points[0, 0, 1])
    xx, xy = int(image_points[1, 0, 0]), int(image_points[1, 0, 1])
    yx, yy = int(image_points[2, 0, 0]), int(image_points[2, 0, 1])
    zx, zy = int(image_points[3, 0, 0]), int(image_points[3, 0, 1])

    height, width = frame.shape[:2]
    for end_pt, color in (((xx, xy), COLOR_AXIS_X), ((yx, yy), COLOR_AXIS_Y), ((zx, zy), COLOR_AXIS_Z)):
        clipped, start, end = cv2.clipLine((0, 0, width, height), (ox, oy), end_pt)
        if clipped:
            cv2.line(frame, start, end, color, 3)

    if 0 <= ox < width and 0 <= oy < height:
        cv2.circle(frame, (ox, oy), 5, COLOR_TRACKER_ORIGIN, -1)
        cv2.putText(frame, "O", (ox + 4, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        cv2.putText(frame, "X", (xx + 4, xy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AXIS_X, 1)
        cv2.putText(frame, "Y", (yx + 4, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AXIS_Y, 1)
        cv2.putText(frame, "Z", (zx + 4, zy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AXIS_Z, 1)

        x, y, z = position[0]
        label = f"{tracker.name} (X:{x:+.3f}, Y:{y:+.3f}, Z:{z:+.3f})"
        cv2.putText(
            frame,
            label,
            (ox + 12, oy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            2,
        )


def draw_tracker_axes(frame, trackers, camera_matrix, dist_coeffs, rvec, tvec):
    """Draw overlays for every tracker with a current pose estimate."""
    active_trackers = [tracker for tracker in trackers if tracker.pose is not None]

    for tracker in active_trackers:
        draw_tracker_overlay(frame, tracker, camera_matrix, dist_coeffs, rvec, tvec)

    if not active_trackers:
        cv2.putText(
            frame,
            "Waiting for object tags...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WAITING,
            2,
        )


def annotate_frame(
    frame,
    tag_dict,
    *,
    last_reference_pose,
    trackers,
    camera_matrix,
    dist_coeffs,
    use_reference_frame: bool = False,
):
    """Draw tag outlines and tracker overlays on a BGR frame."""
    draw_tag_outlines(frame, tag_dict)

    if use_reference_frame:
        if last_reference_pose is None:
            cv2.putText(
                frame,
                "Waiting for reference tag...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_WAITING,
                2,
            )
            return

        rvec, tvec = reference_extrinsics(last_reference_pose)
        draw_reference_axes(frame, camera_matrix, dist_coeffs, rvec, tvec)
        draw_tracker_axes(frame, trackers, camera_matrix, dist_coeffs, rvec, tvec)
        cv2.putText(
            frame,
            "Mode: reference frame",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_MODE,
            2,
        )
        return

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    draw_tracker_axes(frame, trackers, camera_matrix, dist_coeffs, rvec, tvec)
    cv2.putText(
        frame,
        "Mode: camera frame",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_MODE,
        2,
    )
