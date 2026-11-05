"""
Diagnostic KITTI 3D bounding-box viewer.
Save as: visualize_kitti_boxes_diagnostic.py
Run from the project root:
    python visualize_kitti_boxes_diagnostic.py

This script does not overwrite your original visualizer.
"""
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path("data/kitti")
POINT_CLOUD_DIR = ROOT / "training" / "velodyne"
LABEL_DIR = ROOT / "training" / "label_2"
CALIB_DIR = ROOT / "training" / "calib"
FRAME_ID = "000001"

# Show points within this distance of any annotated box, to make inspection easier.
CROP_MARGIN_METERS = 12.0

CLASS_COLORS = {
    "Car": (0.1, 0.9, 0.2),    # green
    "Van": (0.1, 0.6, 1.0),    # blue
    "Truck": (1.0, 0.15, 0.1), # red
}


def load_calibration(path: Path) -> np.ndarray:
    values = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = np.asarray(
                    [float(v) for v in raw.split()], dtype=np.float64
                )
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing Tr_velo_to_cam or R0_rect in {path}")

    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    return rect @ velo_to_cam


def load_labels(path: Path) -> list[dict]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 15 or fields[0] not in CLASS_COLORS:
                continue
            h, w, length = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            labels.append({
                "type": fields[0],
                "dimensions": (h, w, length),
                "location": (x, y, z),
                "yaw": float(fields[14]),
            })
    return labels


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def box_corners_camera(box: dict) -> np.ndarray:
    h, w, length = box["dimensions"]
    x, y, z = box["location"]
    yaw = box["yaw"]

    local = np.array([
        [ length / 2,  length / 2, -length / 2, -length / 2,
          length / 2,  length / 2, -length / 2, -length / 2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [ w / 2, -w / 2, -w / 2, w / 2,
          w / 2, -w / 2, -w / 2, w / 2],
    ], dtype=np.float64)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return (rot @ local + np.array([[x], [y], [z]], dtype=np.float64)).T


def make_box(corners: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return line_set


def main() -> None:
    point_path = POINT_CLOUD_DIR / f"{FRAME_ID}.bin"
    label_path = LABEL_DIR / f"{FRAME_ID}.txt"
    calib_path = CALIB_DIR / f"{FRAME_ID}.txt"

    for p in (point_path, label_path, calib_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                "Check that data/kitti/training contains velodyne, label_2, and calib folders."
            )

    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Unexpected KITTI point-cloud data in {point_path}")
    points = raw.reshape(-1, 4)
    points_lidar = points[:, :3].astype(np.float64)

    lidar_to_camera = load_calibration(calib_path)
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    labels = load_labels(label_path)

    if not labels:
        raise ValueError(f"No supported Car/Van/Truck labels found in {label_path}")

    box_geometries = []
    box_corners_lidar = []
    for i, label in enumerate(labels, start=1):
        corners_cam = box_corners_camera(label)
        corners_lidar = transform_points(corners_cam, camera_to_lidar)
        box_corners_lidar.append(corners_lidar)
        box_geometries.append(make_box(corners_lidar, CLASS_COLORS[label["type"]]))

        h, w, length = label["dimensions"]
        print(
            f"{i}. {label['type']}: H/W/L={h:.2f}/{w:.2f}/{length:.2f}, "
            f"camera location={label['location']}, yaw={label['yaw']:.3f} rad"
        )
        print(
            "   LiDAR box bounds: "
            f"min={np.round(corners_lidar.min(axis=0), 2)}, "
            f"max={np.round(corners_lidar.max(axis=0), 2)}"
        )

    # Keep only points near at least one annotation, making the boxes easier to inspect.
    keep = np.zeros(len(points_lidar), dtype=bool)
    for corners in box_corners_lidar:
        lo = corners.min(axis=0) - CROP_MARGIN_METERS
        hi = corners.max(axis=0) + CROP_MARGIN_METERS
        keep |= np.all((points_lidar >= lo) & (points_lidar <= hi), axis=1)

    display_points = points_lidar[keep]
    print(f"\nLoaded {len(points_lidar):,} points; displaying {len(display_points):,} points near the boxes.")
    print("Box colors: Car = green, Van = blue, Truck = red.")
    print("If a box still contains only points at one end, the next check is frame pairing / box convention.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(display_points)
    if len(display_points):
        refl = points[keep, 3].astype(np.float64)
        span = float(np.ptp(refl))
        gray = (refl - refl.min()) / span if span > 1e-8 else np.zeros_like(refl)
        cloud.colors = o3d.utility.Vector3dVector(np.column_stack((gray, gray, gray)))

    # A small sphere at each transformed box center makes the annotated center visible.
    center_markers = []
    for label, corners in zip(labels, box_corners_lidar):
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
        marker.translate(corners.mean(axis=0))
        marker.paint_uniform_color(CLASS_COLORS[label["type"]])
        center_markers.append(marker)

    geometries = [cloud, *box_geometries, *center_markers]
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"KITTI Diagnostic Boxes - Frame {FRAME_ID}",
        width=1280,
        height=800,
    )


if __name__ == "__main__":
    main()
