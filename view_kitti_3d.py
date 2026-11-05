from pathlib import Path
import numpy as np
import open3d as o3d

ROOT = Path("data/kitti")
FRAME_ID = "000006"
VELODYNE_DIR = ROOT / "training" / "velodyne"
LABEL_DIR = ROOT / "training" / "label_2"
CALIB_DIR = ROOT / "training" / "calib"
VEHICLE_CLASSES = {"Car", "Van", "Truck"}
GREEN = np.array([0.05, 1.0, 0.02], dtype=np.float64)
RED = [1.0, 0.01, 0.01]


def load_lidar(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Invalid or empty LiDAR file: {path}")
    return raw.reshape(-1, 4)


def read_calibration(path):
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = np.asarray([float(v) for v in raw.split()])
            except ValueError:
                pass
    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing calibration values in {path}")
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    return rect @ velo_to_cam


def read_labels(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 15 or fields[0] not in VEHICLE_CLASSES:
                continue
            try:
                h, w, length = map(float, fields[8:11])
                loc = np.array(list(map(float, fields[11:14])), dtype=np.float64)
                yaw = float(fields[14])
            except ValueError:
                continue
            if min(h, w, length) <= 0:
                continue
            result.append({"class": fields[0], "h": h, "w": w, "l": length,
                           "location": loc, "yaw": yaw})
    return result


def transform_points(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (matrix @ homogeneous.T).T[:, :3]


def box_corners_camera(obj):
    h, w, length = obj["h"], obj["w"], obj["l"]
    x, y, z = obj["location"]
    yaw = obj["yaw"]
    local = np.array([
        [length/2, length/2, -length/2, -length/2,
         length/2, length/2, -length/2, -length/2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (rotation @ local + np.array([[x], [y], [z]])).T


def inside_box_camera(points_camera, obj):
    h, w, length = obj["h"], obj["w"], obj["l"]
    c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (rotation.T @ (points_camera - obj["location"]).T).T
    return ((np.abs(local[:, 0]) <= length/2)
            & (local[:, 1] <= 0.15)
            & (local[:, 1] >= -h - 0.15)
            & (np.abs(local[:, 2]) <= w/2))


def make_box(corners_lidar):
    edges = [[0,1], [1,2], [2,3], [3,0],
             [4,5], [5,6], [6,7], [7,4],
             [0,4], [1,5], [2,6], [3,7]]
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(corners_lidar)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(RED), (len(edges), 1)))
    return box


def heat_colors(points_camera, obj):
    center = obj["location"].copy()
    center[1] -= obj["h"] / 2
    distance = np.linalg.norm(points_camera - center, axis=1)
    scale = max(np.sqrt(obj["l"]**2 + obj["w"]**2 + obj["h"]**2) / 2, 1e-6)
    t = np.clip(distance / scale, 0, 1)
    # Near vehicle center = red; toward the box edges = yellow.
    return np.column_stack((np.ones_like(t), 0.95 * t, np.zeros_like(t)))


def main():
    lidar_path = VELODYNE_DIR / f"{FRAME_ID}.bin"
    label_path = LABEL_DIR / f"{FRAME_ID}.txt"
    calib_path = CALIB_DIR / f"{FRAME_ID}.txt"
    for path in (lidar_path, label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Check ROOT and the training paths at the top of this script."
            )

    lidar = load_lidar(lidar_path)
    points_lidar = lidar[:, :3].astype(np.float64)
    lidar_to_camera = read_calibration(calib_path)
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    points_camera = transform_points(points_lidar, lidar_to_camera)
    objects = read_labels(label_path)

    colors = np.tile(GREEN, (len(points_lidar), 1))
    for obj in objects:
        mask = inside_box_camera(points_camera, obj)
        print(f'{obj["class"]}: {int(mask.sum())} points inside box')
        if mask.any():
            colors[mask] = heat_colors(points_camera[mask], obj)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_lidar)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    geometries = [cloud]
    for obj in objects:
        corners_camera = box_corners_camera(obj)
        corners_lidar = transform_points(corners_camera, camera_to_lidar)
        geometries.append(make_box(corners_lidar))

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name=f"KITTI 3D LiDAR Vehicle Viewer - {FRAME_ID}",
        width=1500, height=900, visible=True)
    for geometry in geometries:
        viewer.add_geometry(geometry)

    render = viewer.get_render_option()
    render.background_color = np.array([0.0, 0.0, 0.0])
    render.point_size = 2.0
    render.line_width = 2.0
    render.show_coordinate_frame = False

    view = viewer.get_view_control()
    view.set_lookat([0.0, 0.0, 15.0])
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.12)

    print(f"Frame: {FRAME_ID} | LiDAR points: {len(points_lidar):,}")
    print(f"Vehicle boxes: {len(objects)}")
    print("Green = LiDAR scan | Yellow/red = points inside labeled vehicle boxes")
    print("Red wireframes = KITTI 3D bounding boxes")
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()
