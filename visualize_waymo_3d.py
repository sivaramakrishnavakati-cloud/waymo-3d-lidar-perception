import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Load Waymo data
points = np.load("data/waymo/processed/waymo_frame_000.npy")
labels = pd.read_csv("data/waymo/processed/waymo_labels_000.csv")

# Keep a sample of points for smooth interactive rendering
rng = np.random.default_rng(42)
sample_size = min(30000, len(points))
sample = points[rng.choice(len(points), sample_size, replace=False)]

fig = go.Figure()

# LiDAR point cloud
fig.add_trace(go.Scatter3d(
    x=sample[:, 0],
    y=sample[:, 1],
    z=sample[:, 2],
    mode="markers",
    marker=dict(
        size=1.2,
        color=sample[:, 2],
        colorscale="Viridis",
        opacity=0.8,
        colorbar=dict(title="Height (m)")
    ),
    name="LiDAR Points"
))

# Draw ground-truth 3D bounding boxes
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

vehicle_labels = labels[labels["type"] == 1]

for _, obj in vehicle_labels.iterrows():
    length = obj["length"]
    width = obj["width"]
    height = obj["height"]
    heading = obj["heading"]

    # Local bounding-box corners
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

    # Rotate around Z axis
    rotation = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])

    corners = corners @ rotation.T

    # Translate to object center
    corners += np.array([
        obj["center_x"],
        obj["center_y"],
        obj["center_z"]
    ])

    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[start, 0], corners[end, 0]],
            y=[corners[start, 1], corners[end, 1]],
            z=[corners[start, 2], corners[end, 2]],
            mode="lines",
            line=dict(color="red", width=5),
            showlegend=False,
            hoverinfo="skip"
        ))

fig.update_layout(
    title=dict(
        text="WAYMO OPEN DATASET | 3D PERCEPTION",
        x=0.5,
        font=dict(size=22, color="white")
    ),
    paper_bgcolor="#080D16",
    plot_bgcolor="#080D16",
    font=dict(color="white"),
    scene=dict(
        bgcolor="#080D16",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        zaxis_title="Z (meters)",
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.6, y=-1.6, z=1.1)
        )
    ),
    margin=dict(l=0, r=0, t=65, b=0),
    height=900,
    showlegend=True
)

output = Path("outputs/waymo_3d_visualization.html")
output.parent.mkdir(exist_ok=True)

fig.write_html(
    str(output),
    include_plotlyjs=True,
    full_html=True,
    config={"responsive": True}
)

print("Improved 3D visualization saved:", output.resolve())

# Save interactive visualization
output = Path("outputs/waymo_3d_visualization.html")
output.parent.mkdir(exist_ok=True)

fig.write_html(str(output), include_plotlyjs=True)

print("3D visualization created successfully!")
print("LiDAR points displayed:", sample_size)
print("Ground-truth vehicles:", len(vehicle_labels))
print("Saved to:", output.resolve())