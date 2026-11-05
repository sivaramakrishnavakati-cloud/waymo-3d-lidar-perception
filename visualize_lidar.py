
import numpy as np
import open3d as o3d

print("=" * 50)
print("LiDAR AI/ML Project - 3D Point Cloud Visualization")
print("=" * 50)

# Generate sample 3D points to simulate a LiDAR point cloud
np.random.seed(42)

number_of_points = 5000

points = np.random.uniform(
    low=[-10, -10, -3],
    high=[10, 10, 3],
    size=(number_of_points, 3)
)

# Create the 3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Color the points based on their height
heights = points[:, 2]
normalized_heights = (heights - heights.min()) / (
    heights.max() - heights.min()
)

colors = np.column_stack((
    normalized_heights,
    1 - normalized_heights,
    np.full(number_of_points, 0.5)
))

point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Display the 3D visualization
print(f"Number of points: {len(point_cloud.points)}")
print("Opening 3D visualization window...")

o3d.visualization.draw_geometries(
    [point_cloud],
    window_name="LiDAR 3D Point Cloud",
    width=1000,
    height=700
)

print("LiDAR visualization completed.")

