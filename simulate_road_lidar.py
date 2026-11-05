
import numpy as np
import open3d as o3d

np.random.seed(42)

print("=" * 50)
print("LiDAR AI/ML - Simulated Road Scene")
print("=" * 50)

# 1. Simulate road surface points
road_x = np.random.uniform(-15, 15, 5000)
road_y = np.random.uniform(0, 40, 5000)
road_z = np.random.normal(0, 0.03, 5000)

road = np.column_stack((road_x, road_y, road_z))

# 2. Simulate a vehicle-shaped point cloud
vehicle_x = np.random.uniform(-2, 2, 1800)
vehicle_y = np.random.uniform(12, 16, 1800)
vehicle_z = np.random.uniform(0.3, 2.0, 1800)

vehicle = np.column_stack(
    (vehicle_x, vehicle_y, vehicle_z)
)

# 3. Combine road and vehicle points
points = np.vstack((road, vehicle))

# 4. Color road points gray and vehicle points red
road_colors = np.tile(
    [0.55, 0.55, 0.55], (len(road), 1)
)

vehicle_colors = np.tile(
    [1.0, 0.1, 0.1], (len(vehicle), 1)
)

colors = np.vstack((road_colors, vehicle_colors))

# 5. Create Open3D point cloud
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors)

print(f"Road points: {len(road)}")
print(f"Vehicle points: {len(vehicle)}")
print(f"Total points: {len(points)}")
print("Opening simulated road scene...")

# 6. Visualize
o3d.visualization.draw_geometries(
    [cloud],
    window_name="Simulated LiDAR Road Scene",
    width=1100,
    height=750
)

print("Visualization completed.")

