import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path("data/waymo/processed/multi_frame")

tracks = {}

# Collect vehicle positions across all frames
for frame_idx in range(100):

    file = DATA_DIR / f"frame_{frame_idx:04d}.csv"

    if not file.exists():
        continue

    df = pd.read_csv(file)
    vehicles = df[df["type"] == 1]

    for _, obj in vehicles.iterrows():

        object_id = obj["object_id"]

        if object_id not in tracks:
            tracks[object_id] = []

        tracks[object_id].append({
            "frame": frame_idx,
            "x": obj["center_x"],
            "y": obj["center_y"],
            "z": obj["center_z"]
        })

# Display tracking statistics
print("WAYMO VEHICLE TRACKING")
print("-" * 40)

print("Unique vehicle IDs:", len(tracks))

for object_id, positions in list(tracks.items())[:5]:

    print("\nVehicle ID:", object_id)
    print("Frames tracked:", len(positions))
    print("First position:", positions[0])
    print("Last position:", positions[-1])

# Create interactive trajectory visualization
fig = go.Figure()

for object_id, positions in tracks.items():

    positions = sorted(positions, key=lambda p: p["frame"])

    x = [p["x"] for p in positions]
    y = [p["y"] for p in positions]

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=f"Vehicle {object_id[:8]}",
        line=dict(width=2),
        marker=dict(size=4),
        hovertemplate=(
            "X: %{x:.2f} m<br>"
            "Y: %{y:.2f} m"
            "<extra></extra>"
        )
    ))

fig.update_layout(
    title="Waymo Multi-Object Tracking | Vehicle Trajectories",
    template="plotly_dark",
    xaxis_title="X Position (meters)",
    yaxis_title="Y Position (meters)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=750
)

output = Path("outputs/waymo_vehicle_tracking.html")
output.parent.mkdir(exist_ok=True)

fig.write_html(str(output))

print("\nTracking visualization saved:", output.resolve())