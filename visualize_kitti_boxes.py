from pathlib import Path

import numpy as np
import open3d as o3d


# ==================================================
# Configuration
# ==================================================
ROOT = Path("data/kitti")
POINT_CLOUD_DIR = ROOT / "training" / "velodyne"
LABEL_DIR = ROOT / "training" / "label_2"
CALIB_DIR = ROOT / "training" / "calib"
FRAME_ID = "000003"

SUPPORTED_CLASSES = {"Car", "Van", "Truck"}


# ==================================================
# Calibration
# ==================================================
def load_calibration(calib_path: Path) -> np.ndarray:
    """Return the 4x4 transform from LiDAR to rectified camera coordinates."""
    calibration = {}
    with calib_path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            try:
                calibration[key.strip()] = np.asarray(
                    [float(value) for value in values.split()], dtype=np.float64
                )
            except ValueError:
                continue

    for key in ("Tr_velo_to_cam", "R0_rect"):
        if key not in calibration:
            raise ValueError(f"Missing {key} in calibration file: {calib_path}")

    if calibration["Tr_velo_to_cam"].size != 12:
        raise ValueError("Tr_velo_to_cam must contain 12 values.")
    if calibration["R0_rect"].size != 9:
        raise ValueError("R0_rect must contain 9 values.")

    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = calibration["Tr_velo_to_cam"].reshape(3, 4)

    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = calibration["R0_rect"].reshape(3, 3)

    # KITTI transform order: first Velodyne -> camera, then rectification.
    return rectification @ velo_to_cam


# ==================================================
# Data loading
# ==================================================
def load_points(bin_path: Path) -> np.ndarray:
    """Load KITTI Velodyne points as N x 4: x, y, z, reflectance."""
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Point cloud is empty: {bin_path}")
    if raw.size % 4 != 0:
        raise ValueError(
            f"Unexpected point-cloud format in {bin_path}: number of floats "
            f"({raw.size}) is not divisible by 4."
        )
    return raw.reshape(-1, 4)


def load_labels(label_path: Path) -> list[dict]:
    """Load supported KITTI labels; location is bottom center in rectified camera coordinates."""
    objects = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if len(fields) < 15 or fields[0] not in SUPPORTED_CLASSES:
                continue
            try:
                height, width, length = map(float, fields[8:11])
                x, y, z = map(float, fields[11:14])
                rotation_y = float(fields[14])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label values at {label_path}:{line_number}"
                ) from exc

            objects.append(
                {
                    "type": fields[0],
                    "dimensions": (height, width, length),
                    "location": (x, y, z),
                    "rotation_y": rotation_y,
                }
            )
    return objects


# ==================================================
# Box geometry and diagnostics
# ==================================================
def camera_box_corners(box: dict) -> np.ndarray:
    """Create 8 KITTI box corners in rectified camera coordinates (8 x 3).

    KITTI dimensions are (height, width, length). The annotation location is
    the bottom center. In the camera frame, local X is length, local Y is
    height (positive downward in the camera frame), and local Z is width.
    The top face therefore has camera-local Y = -height relative to the
    bottom-center origin.
    """
    height, width, length = box["dimensions"]
    x, y, z = box["location"]
    yaw = box["rotation_y"]

    # Canonical KITTI ordering: bottom four corners, then top four corners.
    local = np.array(
        [
            [ length / 2,  length / 2, -length / 2, -length / 2,
              length / 2,  length / 2, -length / 2, -length / 2],
            [0, 0, 0, 0, -height, -height, -height, -height],
            [ width / 2, -width / 2, -width / 2, width / 2,
              width / 2, -width / 2, -width / 2, width / 2],
        ],
        dtype=np.float64,
    )

    c, s = np.cos(yaw), np.sin(yaw)
    rotation_y = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )
    rotated = rotation_y @ local
    rotated += np.asarray([[x], [y], [z]], dtype=np.float64)
    return rotated.T


def camera_points_to_lidar(points_camera: np.ndarray, camera_to_lidar: np.ndarray) -> np.ndarray:
    """Transform N x 3 rectified-camera points to N x 3 LiDAR points."""
    points_h = np.column_stack(
        [points_camera, np.ones(len(points_camera), dtype=np.float64)]
    )
    return (camera_to_lidar @ points_h.T).T[:, :3]


def lidar_points_to_camera(points_lidar: np.ndarray, lidar_to_camera: np.ndarray) -> np.ndarray:
    """Transform N x 3 LiDAR points to N x 3 rectified-camera points."""
    points_h = np.column_stack(
        [points_lidar, np.ones(len(points_lidar), dtype=np.float64)]
    )
    return (lidar_to_camera @ points_h.T).T[:, :3]


def count_points_inside_box(points_camera: np.ndarray, box: dict) -> int:
    """Count LiDAR points inside the annotated 3D box, using camera coordinates."""
    height, width, length = box["dimensions"]
    center = np.asarray(box["location"], dtype=np.float64)
    yaw = box["rotation_y"]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation_y = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
    )

    # Undo the box yaw to express points in the box's local coordinate frame.
    local_points = (rotation_y.T @ (points_camera - center).T).T
    inside = (
        (np.abs(local_points[:, 0]) <= length / 2)
        & (local_points[:, 1] <= 0.05)
        & (local_points[:, 1] >= -height - 0.05)
        & (np.abs(local_points[:, 2]) <= width / 2)
    )
    return int(np.count_nonzero(inside))


def create_box_lines(corners: np.ndarray, color=(0.10, 1.00, 0.35)) -> o3d.geometry.LineSet:
    """Create a red wireframe from corners ordered by camera_box_corners()."""
    if corners.shape != (8, 3):
        raise ValueError(f"Expected corners with shape (8, 3), got {corners.shape}")

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector([list(color) for _ in edges])
    return line_set


# ==================================================
# Main
# ==================================================
def main() -> None:
    point_path = POINT_CLOUD_DIR / f"{FRAME_ID}.bin"
    label_path = LABEL_DIR / f"{FRAME_ID}.txt"
    calib_path = CALIB_DIR / f"{FRAME_ID}.txt"

    for path in (point_path, label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Expected data/kitti/training/{velodyne,label_2,calib}."
            )

    print(f"Loading KITTI frame {FRAME_ID}...")
    points = load_points(point_path)
    points_lidar = points[:, :3].astype(np.float64)
    print(f"Loaded {len(points_lidar):,} LiDAR points.")

    lidar_to_rect_camera = load_calibration(calib_path)
    rect_camera_to_lidar = np.linalg.inv(lidar_to_rect_camera)
    labels = load_labels(label_path)

    print("\n--- Calibration sanity check ---")
    identity_check = lidar_to_rect_camera @ rect_camera_to_lidar
    print("Max deviation from identity:", float(np.max(np.abs(identity_check - np.eye(4)))))
    print("LiDAR -> rectified camera matrix:\n", lidar_to_rect_camera)
    print("Rectified camera -> LiDAR matrix:\n", rect_camera_to_lidar)

    # Transform the point cloud only for the diagnostic count. Display the
    # original points in LiDAR coordinates alongside boxes also in LiDAR.
    points_camera = lidar_points_to_camera(points_lidar, lidar_to_rect_camera)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_lidar)



    # ============================================================
    # PROFESSIONAL LIDAR POINT-CLOUD VISUALIZATION
    # ============================================================

    point_count = len(points_lidar)

    # Use LiDAR reflectance/intensity (4th column)
    reflectance = points[:, 3].astype(np.float64)

    # Normalize reflectance safely
    intensity_min = reflectance.min()
    intensity_max = reflectance.max()

    if intensity_max - intensity_min > 1e-8:
        intensity = (reflectance - intensity_min) / (
            intensity_max - intensity_min
        )
    else:
        intensity = np.zeros(point_count)

    # Improve visibility while preserving intensity differences
    intensity = 0.25 + 0.75 * intensity

    # Cool silver-blue point-cloud palette
    colors = np.column_stack((
        0.45 * intensity,
        0.65 * intensity,
        0.90 * intensity
    ))

    cloud.colors = o3d.utility.Vector3dVector(colors)

    geometries = [cloud]



    print(f"\nSupported annotated vehicles: {len(labels)}")
    for index, label in enumerate(labels, start=1):
        corners_camera = camera_box_corners(label)
        corners_lidar = camera_points_to_lidar(corners_camera, rect_camera_to_lidar)
        point_count = count_points_inside_box(points_camera, label)
        dimensions = label["dimensions"]

        print(f"\nBox {index}: {label['type']}")
        print(f"  Camera bottom-center: {label['location']}")
        print(f"  Dimensions H/W/L: {dimensions}")
        print(f"  Rotation Y: {label['rotation_y']:.4f} rad")
        print(f"  LiDAR corner min: {corners_lidar.min(axis=0)}")
        print(f"  LiDAR corner max: {corners_lidar.max(axis=0)}")
        print(f"  LiDAR points inside annotated box: {point_count}")
        if point_count == 0:
            print("  WARNING: no points fall inside this annotated box; inspect calibration, label/frame pairing, and yaw convention.")

        # Bright green vehicle bounding boxes
        geometries.append(
            create_box_lines(
                corners_lidar,
                color=(0.10, 1.00, 0.30)
            )
        )

    # Add a coordinate frame at the LiDAR origin for orientation reference.
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0))

    print("\nOpening 3D viewer. Drag to rotate and scroll to zoom.")
    print("Close the viewer window to end the script.")

    # Professional visualization settings
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=f"Automotive LiDAR 3D Perception - Frame {FRAME_ID}",
        width=1600,
        height=900
    )

    # Display the LiDAR point cloud
    visualizer.add_geometry(cloud)

    # Display the bounding boxes already created by your script.
    # Skip the coordinate-frame geometry to remove the large red arrow.
    for geometry in geometries[1:]:
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            continue
        visualizer.add_geometry(geometry)

    # Dark background and clean point rendering
    render_option = visualizer.get_render_option()
    render_option.background_color = np.array([0.035, 0.045, 0.065])
    render_option.point_size = 2.0
    render_option.show_coordinate_frame = False

    # Open the visualization window
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
