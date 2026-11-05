from pathlib import Path
import numpy as np
import open3d as o3d

ROOT = Path("data/kitti")
FRAME = "000001"
POINT = ROOT / "training" / "velodyne" / f"{FRAME}.bin"
LABEL = ROOT / "training" / "label_2" / f"{FRAME}.txt"
CALIB = ROOT / "training" / "calib" / f"{FRAME}.txt"
CLASSES = {"Car": (0.1, 0.9, 0.2), "Van": (0.1, 0.6, 1.0), "Truck": (1.0, 0.15, 0.1)}

def homog(p):
    return np.column_stack((p, np.ones(len(p))))

def read_calib(path):
    d = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        try: d[k.strip()] = np.array([float(x) for x in v.split()], dtype=float)
        except ValueError: pass
    if "Tr_velo_to_cam" not in d or "R0_rect" not in d:
        raise ValueError(f"Missing calibration fields in {path}")
    t = np.eye(4); t[:3, :] = d["Tr_velo_to_cam"].reshape(3, 4)
    r = np.eye(4); r[:3, :3] = d["R0_rect"].reshape(3, 3)
    return r @ t

def read_labels(path):
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        f = line.split()
        if len(f) < 15 or f[0] not in CLASSES: continue
        h, w, l = map(float, f[8:11])
        xyz = np.array(list(map(float, f[11:14])), dtype=float)
        out.append({"type": f[0], "h": h, "w": w, "l": l, "xyz": xyz, "yaw": float(f[14]), "line": line_no})
    return out

def corners_cam(b):
    h, w, l = b["h"], b["w"], b["l"]
    local = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                      [0,0,0,0,-h,-h,-h,-h],
                      [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]], dtype=float)
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
    rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return (rot @ local + b["xyz"][:, None]).T

def lineset(corners, color):
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls

def main():
    for p in (POINT, LABEL, CALIB):
        if not p.is_file(): raise FileNotFoundError(f"Missing required file: {p}")
    raw = np.fromfile(POINT, dtype=np.float32)
    if raw.size == 0 or raw.size % 4: raise ValueError(f"Invalid KITTI .bin: {POINT}")
    raw = raw.reshape(-1, 4)
    lidar = raw[:, :3].astype(float)
    l2c = read_calib(CALIB); c2l = np.linalg.inv(l2c)
    cam_points = (l2c @ homog(lidar).T).T[:, :3]
    boxes = read_labels(LABEL)
    print(f"Frame {FRAME}: {len(lidar):,} LiDAR points; {len(boxes)} supported labels.")
    print("LiDAR axes: X forward, Y left, Z up. Camera axes: X right, Y down, Z forward.")
    print(f"Transform inverse-consistency error: {np.max(np.abs(l2c @ c2l - np.eye(4))):.3e}")
    geoms, all_corners = [], []
    for i, b in enumerate(boxes, 1):
        cc = corners_cam(b)
        cl = (c2l @ homog(cc).T).T[:, :3]
        center_lidar = (c2l @ np.r_[b["xyz"], 1.0])[:3]
        dist = np.linalg.norm(lidar - center_lidar, axis=1)
        print(f"\n{i}. {b['type']} (label line {b['line']})")
        print(f"  Camera location={np.round(b['xyz'],3)}, yaw={b['yaw']:.3f} rad, H/W/L={b['h']:.2f}/{b['w']:.2f}/{b['l']:.2f}")
        print(f"  Transformed LiDAR center={np.round(center_lidar,3)}")
        print(f"  Box bounds min={np.round(cl.min(axis=0),3)}, max={np.round(cl.max(axis=0),3)}")
        print(f"  Points within 3m/6m of center: {int((dist<=3).sum())}/{int((dist<=6).sum())}")
        print(f"  Nearest 10 point distances (m): {np.round(np.sort(dist)[:10],2)}")
        all_corners.append(cl)
    near = np.zeros(len(lidar), dtype=bool)
    for cl in all_corners:
        near |= np.all((lidar >= cl.min(axis=0)-10) & (lidar <= cl.max(axis=0)+10), axis=1)
    shown = lidar[near]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(shown)
    if len(shown):
        refl = raw[near, 3].astype(float); span = np.ptp(refl)
        gray = (refl-refl.min())/span if span > 1e-8 else np.zeros_like(refl)
        cloud.colors = o3d.utility.Vector3dVector(np.column_stack((gray, gray, gray)))
    geoms.append(cloud)
    for b, cl in zip(boxes, all_corners):
        geoms.append(lineset(cl, CLASSES[b["type"]]))
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.22)
        marker.translate(cl.mean(axis=0)); marker.paint_uniform_color(CLASSES[b["type"]]); geoms.append(marker)
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        axis_cam = np.vstack((b["xyz"], b["xyz"] + np.array([c, 0, -s])*3))
        axis_lidar = (c2l @ homog(axis_cam).T).T[:, :3]
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(axis_lidar)
        axis.lines = o3d.utility.Vector2iVector([[0,1]])
        axis.colors = o3d.utility.Vector3dVector([(1.0,0.8,0.0)])
        geoms.append(axis)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    print(f"\nViewer displays {len(shown):,} nearby points. Yellow line shows box length direction.")
    print("This script is diagnostic only; it does not change your data.")
    o3d.visualization.draw_geometries(geoms, window_name=f"KITTI Geometry Validation - {FRAME}", width=1280, height=800)

if __name__ == "__main__":
    main()
