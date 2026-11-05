"""
KITTI box geometry diagnostic (does not overwrite your original script).

Run from the project root:
    python diagnose_kitti_box_geometry.py

It compares standard KITTI yaw against an alternative yaw sign for diagnostics only.
It also reports points inside each box and prints projected box bounds. Do not treat
the alternative as correct merely because it has a larger point count.
"""
from pathlib import Path
import numpy as np

ROOT = Path("data/kitti")
POINT_CLOUD_DIR = ROOT / "training" / "velodyne"
LABEL_DIR = ROOT / "training" / "label_2"
CALIB_DIR = ROOT / "training" / "calib"
FRAME_ID = "000001"
SUPPORTED_CLASSES = {"Car", "Van", "Truck"}
TOLERANCE = 0.05


def read_calibration(path: Path) -> np.ndarray:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            data[key.strip()] = np.array([float(x) for x in raw.split()], dtype=np.float64)
        except ValueError:
            pass
    if "Tr_velo_to_cam" not in data or "R0_rect" not in data:
        raise ValueError(f"Missing calibration entries in {path}")
    velo = np.eye(4)
    velo[:3, :] = data["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = data["R0_rect"].reshape(3, 3)
    return rect @ velo


def read_labels(path: Path) -> list[dict]:
    labels = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 15 or fields[0] not in SUPPORTED_CLASSES:
            continue
        try:
            h, w, length = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            yaw = float(fields[14])
        except ValueError as exc:
            raise ValueError(f"Invalid label line {line_no}") from exc
        labels.append({
            "type": fields[0], "h": h, "w": w, "l": length,
            "center": np.array([x, y, z], dtype=np.float64), "yaw": yaw,
        })
    return labels


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.column_stack((points, np.ones(len(points))))


def corners_camera(box: dict, yaw: float) -> np.ndarray:
    h, w, length = box["h"], box["w"], box["l"]
    local = np.array([
        [ length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [ w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return (rot @ local + box["center"][:, None]).T


def points_inside_camera_box(points_camera: np.ndarray, box: dict, yaw: float) -> int:
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    local = (rot.T @ (points_camera - box["center"]).T).T
    inside = (
        (np.abs(local[:, 0]) <= box["l"] / 2 + TOLERANCE)
        & (local[:, 1] <= TOLERANCE)
        & (local[:, 1] >= -box["h"] - TOLERANCE)
        & (np.abs(local[:, 2]) <= box["w"] / 2 + TOLERANCE)
    )
    return int(inside.sum())


def main() -> None:
    point_path = POINT_CLOUD_DIR / f"{FRAME_ID}.bin"
    label_path = LABEL_DIR / f"{FRAME_ID}.txt"
    calib_path = CALIB_DIR / f"{FRAME_ID}.txt"
    for path in (point_path, label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing: {path}\nExpected data/kitti/training/{{velodyne,label_2,calib}}."
            )

    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Invalid KITTI .bin file: {point_path}")
    points_lidar = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    lidar_to_camera = read_calibration(calib_path)
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    points_camera = (lidar_to_camera @ to_homogeneous(points_lidar).T).T[:, :3]
    labels = read_labels(label_path)

    print(f"Frame: {FRAME_ID}")
    print(f"Loaded LiDAR points: {len(points_lidar):,}")
    print(f"Supported boxes: {len(labels)}")
    print(f"Calibration inverse max error: {np.max(np.abs(lidar_to_camera @ camera_to_lidar - np.eye(4))):.3e}")
    print("\nPoint count comparison (diagnostic only):")

    for i, box in enumerate(labels, 1):
        yaw = box["yaw"]
        count_standard = points_inside_camera_box(points_camera, box, yaw)
        count_flipped = points_inside_camera_box(points_camera, box, -yaw)
        standard_corners = (camera_to_lidar @ to_homogeneous(corners_camera(box, yaw)).T).T[:, :3]
        flipped_corners = (camera_to_lidar @ to_homogeneous(corners_camera(box, -yaw)).T).T[:, :3]

        print(f"\n{i}. {box['type']}")
        print(f"   location (camera): {np.round(box['center'], 3)}")
        print(f"   H/W/L: {box['h']:.2f}/{box['w']:.2f}/{box['l']:.2f}")
        print(f"   label yaw: {yaw:.4f} rad")
        print(f"   points inside standard KITTI yaw: {count_standard}")
        print(f"   points inside reversed-yaw candidate: {count_flipped}")
        print(f"   standard box LiDAR min: {np.round(standard_corners.min(axis=0), 3)}")
        print(f"   standard box LiDAR max: {np.round(standard_corners.max(axis=0), 3)}")
        print(f"   reversed box LiDAR min: {np.round(flipped_corners.min(axis=0), 3)}")
        print(f"   reversed box LiDAR max: {np.round(flipped_corners.max(axis=0), 3)}")

    print("\nImportant: point counts alone cannot establish the correct yaw.")
    print("Both candidates use the same annotation center and dimensions; use the")
    print("reported geometry and visualization to diagnose alignment, not to choose")
    print("whichever candidate happens to contain more points.")


if __name__ == "__main__":
    main()
