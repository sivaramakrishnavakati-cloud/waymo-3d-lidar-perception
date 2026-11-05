"""
KITTI box inlier visualization diagnostic.
Run from project root: python visualize_kitti_box_inliers.py
Does not modify existing scripts or data.
"""
from pathlib import Path
import numpy as np
import open3d as o3d

ROOT = Path("data/kitti")
FRAME_ID = "000001"
POINT_DIR = ROOT / "training" / "velodyne"
LABEL_DIR = ROOT / "training" / "label_2"
CALIB_DIR = ROOT / "training" / "calib"
CLASSES = {"Car": (0.1, 0.9, 0.2), "Van": (0.1, 0.6, 1.0), "Truck": (1.0, 0.15, 0.1)}
INLIER = {"Car": (0.0, 1.0, 1.0), "Van": (1.0, 0.0, 1.0), "Truck": (1.0, 0.8, 0.0)}
TOL = 0.05

def homogeneous(p):
    return np.column_stack((p, np.ones(len(p))))

def load_calib(path):
    d = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        try: d[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
        except ValueError: pass
    if "Tr_velo_to_cam" not in d or "R0_rect" not in d:
        raise ValueError(f"Missing calibration entries in {path}")
    tr = np.eye(4); tr[:3, :] = d["Tr_velo_to_cam"].reshape(3, 4)
    r = np.eye(4); r[:3, :3] = d["R0_rect"].reshape(3, 3)
    return r @ tr

def load_labels(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 15 or f[0] not in CLASSES: continue
        h, w, length = map(float, f[8:11])
        x, y, z = map(float, f[11:14])
        out.append({"type": f[0], "h": h, "w": w, "l": length,
                    "center": np.array([x, y, z]), "yaw": float(f[14])})
    return out

def corners_cam(b):
    h,w,l = b["h"], b["w"], b["l"]
    local = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                      [0,0,0,0,-h,-h,-h,-h],
                      [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]], dtype=float)
    c,s = np.cos(b["yaw"]), np.sin(b["yaw"])
    rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return (rot @ local + b["center"][:,None]).T

def inside_mask(points_cam, b):
    c,s = np.cos(b["yaw"]), np.sin(b["yaw"])
    rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    local = (rot.T @ (points_cam-b["center"]).T).T
    return ((np.abs(local[:,0]) <= b["l"]/2+TOL) &
            (local[:,1] <= TOL) & (local[:,1] >= -b["h"]-TOL) &
            (np.abs(local[:,2]) <= b["w"]/2+TOL))

def make_lines(corners, color):
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls

def main():
    pp = POINT_DIR / f"{FRAME_ID}.bin"
    lp = LABEL_DIR / f"{FRAME_ID}.txt"
    cp = CALIB_DIR / f"{FRAME_ID}.txt"
    for p in (pp, lp, cp):
        if not p.is_file(): raise FileNotFoundError(f"Missing file: {p}")
    raw = np.fromfile(pp, dtype=np.float32)
    if raw.size == 0 or raw.size % 4: raise ValueError(f"Invalid point cloud: {pp}")
    raw = raw.reshape(-1,4)
    lidar = raw[:,:3].astype(float)
    l2c = load_calib(cp); c2l = np.linalg.inv(l2c)
    cam = (l2c @ homogeneous(lidar).T).T[:,:3]
    labels = load_labels(lp)
    corners_list, masks = [], []
    for b in labels:
        cc = corners_cam(b)
        cl = (c2l @ homogeneous(cc).T).T[:,:3]
        corners_list.append(cl)
        masks.append(inside_mask(cam,b))
    near = np.zeros(len(lidar), dtype=bool)
    for cl in corners_list:
        near |= np.all((lidar >= cl.min(axis=0)-12) & (lidar <= cl.max(axis=0)+12), axis=1)
    basepts = lidar[near]
    base = o3d.geometry.PointCloud()
    base.points = o3d.utility.Vector3dVector(basepts)
    if len(basepts):
        refl = raw[near,3].astype(float); span = np.ptp(refl)
        g = (refl-refl.min())/span if span > 1e-8 else np.zeros_like(refl)
        base.colors = o3d.utility.Vector3dVector(np.column_stack((g,g,g)))
    geoms = [base]
    print(f"Frame {FRAME_ID}: {len(lidar):,} total points; displaying {len(basepts):,} nearby points.")
    for i,(b,cl,mask) in enumerate(zip(labels,corners_list,masks),1):
        count = int(mask.sum())
        print(f"{i}. {b['type']}: {count} points inside box; LiDAR bounds min={np.round(cl.min(axis=0),2)}, max={np.round(cl.max(axis=0),2)}")
        geoms.append(make_lines(cl, CLASSES[b["type"]]))
        pts = lidar[mask]
        if len(pts):
            hi = o3d.geometry.PointCloud()
            hi.points = o3d.utility.Vector3dVector(pts)
            hi.paint_uniform_color(INLIER[b["type"]])
            geoms.append(hi)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
        marker.translate(cl.mean(axis=0)); marker.paint_uniform_color(CLASSES[b["type"]])
        geoms.append(marker)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    print("Box colors: Truck=red, Car=green. In-box points: Truck=yellow, Car=cyan.")
    print("Rotate and zoom around the boxes; no source files are modified.")
    o3d.visualization.draw_geometries(geoms, window_name=f"KITTI Inlier Diagnostic - {FRAME_ID}", width=1280, height=800)

if __name__ == "__main__":
    main()
