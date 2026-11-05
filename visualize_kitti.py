
from pathlib import Path

import numpy as np
import open3d as o3d

# Project dataset directory
PROJECT_DIR = Path(__file__).resolve().parent
LIDAR_DIR = PROJECT_DIR / "data" / "kitti" / "training" / "velodyne"

# Load the first LiDAR point cloud
point_cloud_file = LIDAR_DIR / "000000.bin"

if not point_cloud_file.exists():
    raise FileNotFoundError(
        f"LiDAR file not found: {point_cloud_file}"
    )

# KITTI stores each point as x, y, z, reflectance
points = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)

xyz = points[:, :3]
reflectance = points[:, 3]

print("=" * 55)
print("KITTI REAL LiDAR POINT CLOUD")
print("=" * 55)
print(f"File: {point_cloud_file.name}")
print(f"Number of points: {len(points):,}")
print(f"X range: {xyz[:, 0].min():.2f} to {xyz[:, 0].max():.2f} m")
print(f"Y range: {xyz[:, 1].min():.2f} to {xyz[:, 1].max():.2f} m")
print(f"Z range: {xyz[:, 2].min():.2f} to {xyz[:, 2].max():.2f} m")
print(f"Mean reflectance: {reflectance.mean():.4f}")

# Create an Open3D point cloud
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)

# Color points using height for easier visual interpretation
z_min = xyz[:, 2].min()
z_max = xyz[:, 2].max()
z_normalized = (xyz[:, 2] - z_min) / max(z_max - z_min, 1e-6)

colors = np.column_stack(
    (z_normalized, 0.3 * np.ones_like(z_normalized), 1 - z_normalized)
)
cloud.colors = o3d.utility.Vector3dVector(colors)

print("\nOpening 3D visualization...")
o3d.visualization.draw_geometries(
    [cloud],
    window_name="Real KITTI LiDAR Point Cloud",
    width=1200,
    height=800
)
