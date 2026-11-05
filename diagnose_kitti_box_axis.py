"""
KITTI box-axis point distribution diagnostic.
Run from the project root:
    python diagnose_kitti_box_axis.py

Uses the training frame 000001 and does not modify any source files.
"""
from pathlib import Path
import numpy as np

ROOT = Path("data/kitti")
FRAME = "000001"
POINT_PATH = ROOT / "training" / "velodyne" / f"{FRAME}.bin"
LABEL_PATH = ROOT / "training" / "label_2" / f"{FRAME}.txt"
CALIB_PATH = ROOT / "training" / "calib" / f"{FRAME}.txt"
CLASSES = {"Car", "Van", "Truck"}


def read_calibration(path):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            data[key.strip()] = np.asarray([float(v) for v in value.split()], dtype=np.float64)
        except ValueError:
            pass
    if "Tr_velo_to_cam" not in data or "R0_rect" not in data:
        raise ValueError(f"Missing calibration fields in {path}")
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = data["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = data["R0_rect"].reshape(3, 3)
    return rect @ velo_to_cam


def read_boxes(path):
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        f = line.split()
        if len(f) < 15 or f[0] not in CLASSES:
            continue
        h, w, length = map(float, f[8:11])
        loc = np.asarray(list(map(float, f[11:14])), dtype=np.float64)
        boxes.append({
            "class": f[0], "line": line_no, "h": h, "w": w, "l": length,
            "location": loc, "yaw": float(f[14])
        })
    return boxes


def homogeneous(points):
    return np.column_stack((points, np.ones(len(points), dtype=np.float64)))


def main():
    for path in (POINT_PATH, LABEL_PATH, CALIB_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    raw = np.fromfile(POINT_PATH, dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Invalid point-cloud file: {POINT_PATH}")
    raw = raw.reshape(-1, 4)
    lidar = raw[:, :3].astype(np.float64)
    lidar_to_cam = read_calibration(CALIB_PATH)
    cam_to_lidar = np.linalg.inv(lidar_to_cam)
    boxes = read_boxes(LABEL_PATH)

    print(f"Frame: {FRAME}")
    print(f"LiDAR points: {len(lidar):,}")
    print("Axis-bin counts are computed only for points inside each reported box.")
    print("KITTI label dimensions are H, W, L; location is the bottom center in camera coordinates.")

    for index, box in enumerate(boxes, 1):
        # Transform LiDAR points to rectified camera coordinates, then test in box-local coordinates.
        camera_points = (lidar_to_cam @ homogeneous(lidar).T).T[:, :3]
        delta = camera_points - box["location"]
        c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        local = (rot.T @ delta.T).T

        inside = (
            (np.abs(local[:, 0]) <= box["l"] / 2) &
            (local[:, 1] <= 0.05) &
            (local[:, 1] >= -box["h"] - 0.05) &
            (np.abs(local[:, 2]) <= box["w"] / 2)
        )
        points = local[inside]
        print(f"\n{index}. {box['class']} (label line {box['line']})")
        print(f"  Label H/W/L: {box['h']:.2f} / {box['w']:.2f} / {box['l']:.2f} m")
        print(f"  Camera location: {np.round(box['location'], 3)}; yaw: {box['yaw']:.3f} rad")
        print(f"  Points inside standard box: {len(points)}")

        if len(points) == 0:
            print("  No points inside; axis distribution cannot be evaluated.")
            continue

        # Histogram along each local axis; values are descriptive, not a yaw selector.
        bins = [
            ("length axis X", local[inside, 0], box["l"], 8),
            ("vertical axis Y", local[inside, 1], box["h"], 5),
            ("width axis Z", local[inside, 2], box["w"], 6),
        ]
        for axis_name, vals, extent, n_bins in bins:
            counts, edges = np.histogram(vals, bins=n_bins, range=(-extent/2, extent/2))
            print(f"  {axis_name} bin counts: {counts.tolist()}")

        # Counts in each half of the box's length axis.
        length_vals = points[:, 0]
        left_half = int(np.count_nonzero(length_vals < 0))
        right_half = int(np.count_nonzero(length_vals >= 0))
        print(f"  Length-axis halves (negative/positive): {left_half} / {right_half}")
        print(f"  Local point bounds [min,max]: {np.round(points.min(axis=0), 3)} / {np.round(points.max(axis=0), 3)}")
        print("  Interpretation caution: uneven bins can be normal due to occlusion and LiDAR sampling.")

    print("\nDiagnostic complete. This script does not alter any data or select a yaw automatically.")


if __name__ == "__main__":
    main()
