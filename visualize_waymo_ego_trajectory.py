
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Load extracted Waymo poses
POSE_FILE = Path("data/waymo/processed/waymo_poses.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(POSE_FILE)

# Extract vehicle position from each 4x4 transformation matrix
x = df["pose_03"].to_numpy()
y = df["pose_13"].to_numpy()
z = df["pose_23"].to_numpy()

# Shift trajectory so the starting position is the origin
x = x - x[0]
y = y - y[0]
z = z - z[0]

frames = df["frame"].to_numpy()

# Create interactive 3D trajectory
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="lines+markers",
    marker=dict(
        size=4,
        color=frames,
        colorscale="Turbo",
        colorbar=dict(title="Frame"),
    ),
    line=dict(width=5, color="cyan"),
    name="Ego Vehicle Trajectory"
))

# Starting position
fig.add_trace(go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]],
    mode="markers+text",
    marker=dict(size=8, color="green"),
    text=["START"],
    textposition="top center",
    name="Start"
))

# Ending position
fig.add_trace(go.Scatter3d(
    x=[x[-1]], y=[y[-1]], z=[z[-1]],
    mode="markers+text",
    marker=dict(size=8, color="red"),
    text=["END"],
    textposition="top center",
    name="End"
))

fig.update_layout(
    title="Waymo Ego Vehicle Trajectory | 100 Frames",
    scene=dict(
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        zaxis_title="Z Position (m)",
        aspectmode="data"
    ),
    template="plotly_dark",
    height=800
)

output_file = OUTPUT_DIR / "waymo_ego_trajectory.html"
fig.write_html(str(output_file))

print("Trajectory visualization completed!")
print(f"Frames: {len(df)}")
print(f"Start position: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
print(f"End position: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")
print(f"Saved to: {output_file.resolve()}")

fig.show()

