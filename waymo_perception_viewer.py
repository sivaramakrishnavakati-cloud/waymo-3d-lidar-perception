import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Project paths
DATA_DIR = Path("data/waymo/processed/multi_frame")
OUTPUT_DIR = Path("outputs")

POSE_FILE = Path("data/waymo/processed/waymo_poses.csv")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Visualization settings
MAX_POINTS = 30000
FRAME_COUNT = 100

print("Waymo Perception Viewer")
print("Data directory:", DATA_DIR)
print("Pose file:", POSE_FILE)

# Load vehicle poses
poses = pd.read_csv(POSE_FILE)

# Check LiDAR and annotation files
lidar_files = sorted(DATA_DIR.glob("frame_*.npy"))
label_files = sorted(DATA_DIR.glob("frame_*.csv"))

print("\nDATASET VERIFICATION")
print("--------------------")
print("LiDAR frames:", len(lidar_files))
print("Annotation files:", len(label_files))
print("Pose records:", len(poses))

# Verify matching files
assert len(lidar_files) == FRAME_COUNT, "Unexpected LiDAR frame count"
assert len(label_files) == FRAME_COUNT, "Unexpected annotation count"
assert len(poses) == FRAME_COUNT, "Unexpected pose count"

# Load one sample frame
sample_points = np.load(lidar_files[0])
sample_labels = pd.read_csv(label_files[0])

print("\nFIRST FRAME")
print("-----------")
print("Point cloud shape:", sample_points.shape)
print("Vehicle annotations:", len(sample_labels))
print("Annotation columns:", sample_labels.columns.tolist())

print("\nDataset verification successful!")

def get_pose_matrix(frame_index):
    """Return the 4x4 vehicle-to-world transformation matrix."""
    row = poses.iloc[frame_index]

    matrix = np.array([
        [row[f"pose_{i}{j}"] for j in range(4)]
        for i in range(4)
    ])

    return matrix


def transform_points(points, pose_matrix):
    """Transform 3D points into world coordinates."""
    points_h = np.column_stack([
        points,
        np.ones(len(points))
    ])

    transformed = (pose_matrix @ points_h.T).T

    return transformed[:, :3]


# Test transformation on the first frame
pose_matrix = get_pose_matrix(0)

world_points = transform_points(
    sample_points[:1000],
    pose_matrix
)

print("\nCOORDINATE TRANSFORMATION TEST")
print("Pose matrix shape:", pose_matrix.shape)
print("Original points:", sample_points[:1000].shape)
print("Transformed points:", world_points.shape)
print("Transformation successful!")

def get_box_corners(row):
    """Generate the 8 corners of a vehicle's 3D bounding box."""

    length = row["length"]
    width = row["width"]
    height = row["height"]
    heading = row["heading"]

    # Bounding box corners in local coordinates
    corners = np.array([
        [-length/2, -width/2, -height/2],
        [ length/2, -width/2, -height/2],
        [ length/2,  width/2, -height/2],
        [-length/2,  width/2, -height/2],
        [-length/2, -width/2,  height/2],
        [ length/2, -width/2,  height/2],
        [ length/2,  width/2,  height/2],
        [-length/2,  width/2,  height/2]
    ])

    # Rotate around vertical axis
    rotation = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])

    corners = corners @ rotation.T

    # Move to vehicle coordinates
    center = np.array([
        row["center_x"],
        row["center_y"],
        row["center_z"]
    ])

    return corners + center


# Load frame 0
frame_index = 0

points = np.load(DATA_DIR / f"frame_{frame_index:04d}.npy")
labels = pd.read_csv(DATA_DIR / f"frame_{frame_index:04d}.csv")

# Reduce point count for interactive rendering
if len(points) > MAX_POINTS:
    indices = np.random.default_rng(42).choice(
        len(points), MAX_POINTS, replace=False
    )
    points = points[indices]

# Transform LiDAR points into world coordinates
pose = get_pose_matrix(frame_index)
world_points = transform_points(points, pose)

# Center scene relative to the ego vehicle
origin = pose[:3, 3]
world_points -= origin

fig = go.Figure()

# LiDAR point cloud
fig.add_trace(go.Scatter3d(
    x=world_points[:, 0],
    y=world_points[:, 1],
    z=world_points[:, 2],
    mode="markers",
    marker=dict(
        size=1.5,
        color=world_points[:, 2],
        colorscale="Turbo",
        opacity=0.8
    ),
    name="LiDAR Point Cloud"
))

# Ground-truth vehicle bounding boxes
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

for _, row in labels.iterrows():

    if "type" in labels.columns and row["type"] != 1:
        continue

    corners = get_box_corners(row)
    world_corners = transform_points(corners, pose)
    world_corners -= origin

    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[world_corners[start, 0], world_corners[end, 0]],
            y=[world_corners[start, 1], world_corners[end, 1]],
            z=[world_corners[start, 2], world_corners[end, 2]],
            mode="lines",
            line=dict(color="red", width=5),
            showlegend=False,
            hoverinfo="skip"
        ))

fig.update_layout(
    title="Waymo 3D Perception | Frame 0",
    template="plotly_dark",
    scene=dict(
        xaxis_title="World X (m)",
        yaxis_title="World Y (m)",
        zaxis_title="World Z (m)",
        aspectmode="data"
    ),
    height=850
)

output_file = OUTPUT_DIR / "waymo_combined_frame_000.html"
fig.write_html(str(output_file))

print("\nCombined 3D visualization saved:")
print(output_file.resolve())

fig.show()