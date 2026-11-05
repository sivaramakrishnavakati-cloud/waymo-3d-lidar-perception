import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# -------------------------------
# PROJECT SETTINGS
# -------------------------------

DATA_DIR = Path("data/waymo/processed/multi_frame")
POSE_FILE = Path("data/waymo/processed/waymo_poses.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_COUNT = 100
MAX_POINTS = 15000
# Bird's-Eye View settings
BEV_X_RANGE = (-60, 60)
BEV_Y_RANGE = (-40, 40)
BEV_MAX_POINTS = 12000
EGO_MARKER_SIZE = 14
EGO_MARKER_COLOR = "white"

# 3D bounding-box edges
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


# -------------------------------
# LOAD POSES
# -------------------------------

poses = pd.read_csv(POSE_FILE)

lidar_files = sorted(DATA_DIR.glob("frame_*.npy"))
label_files = sorted(DATA_DIR.glob("frame_*.csv"))

assert len(lidar_files) == FRAME_COUNT, "Incorrect LiDAR frame count"
assert len(label_files) == FRAME_COUNT, "Incorrect annotation count"
assert len(poses) == FRAME_COUNT, "Incorrect pose count"


def get_pose_matrix(frame_index):
    row = poses.iloc[frame_index]

    return np.array([
        [row[f"pose_{i}{j}"] for j in range(4)]
        for i in range(4)
    ])


# -------------------------------
# COORDINATE TRANSFORMATION
# -------------------------------

def transform_points(points, matrix):
    points_h = np.column_stack([
        points,
        np.ones(len(points))
    ])

    return (matrix @ points_h.T).T[:, :3]


# -------------------------------
# VEHICLE BOX CORNERS
# -------------------------------

def get_box_corners(row):

    length = row["length"]
    width = row["width"]
    height = row["height"]
    heading = row["heading"]

    corners = np.array([
        [ length/2,  width/2,  height/2],
        [ length/2, -width/2,  height/2],
        [-length/2, -width/2,  height/2],
        [-length/2,  width/2,  height/2],
        [ length/2,  width/2, -height/2],
        [ length/2, -width/2, -height/2],
        [-length/2, -width/2, -height/2],
        [-length/2,  width/2, -height/2]
    ])

    rotation = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])

    corners = corners @ rotation.T

    center = np.array([
        row["center_x"],
        row["center_y"],
        row["center_z"]
    ])

    return corners + center


# -------------------------------
# PREPARE ANIMATION FRAMES
# -------------------------------

print("Preparing Waymo 3D animation...")

animation_frames = []

for frame_index in range(FRAME_COUNT):

    print(f"Processing frame {frame_index + 1}/{FRAME_COUNT}")

    # Load LiDAR
    points = np.load(lidar_files[frame_index])

    # Downsample for smooth browser rendering
    rng = np.random.default_rng(42 + frame_index)

    if len(points) > MAX_POINTS:
        indices = rng.choice(
            len(points),
            MAX_POINTS,
            replace=False
        )
        points = points[indices]

    # Transform into world coordinates
    pose = get_pose_matrix(frame_index)

    ego_origin = pose[:3, 3]

    world_points = transform_points(points, pose)
    world_points -= ego_origin
    world_points = (pose[:3, :3].T @ world_points.T).T

    # Load ground-truth annotations
    labels = pd.read_csv(label_files[frame_index])

    box_x = []
    box_y = []
    box_z = []

    for _, row in labels.iterrows():

        if "type" in labels.columns and row["type"] != 1:
            continue

        corners = get_box_corners(row)

        world_corners = transform_points(corners, pose)
        world_corners -= ego_origin
        world_corners = (pose[:3, :3].T @ world_corners.T).T

        for start, end in EDGES:

            box_x.extend([
                world_corners[start, 0],
                world_corners[end, 0],
                None
            ])

            box_y.extend([
                world_corners[start, 1],
                world_corners[end, 1],
                None
            ])

            box_z.extend([
                world_corners[start, 2],
                world_corners[end, 2],
                None
            ])

    animation_frames.append(
        go.Frame(
            name=str(frame_index),
            data=[
                go.Scatter3d(
                    x=world_points[:, 0],
                    y=world_points[:, 1],
                    z=world_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=1.5,
                        color=world_points[:, 2],
                        colorscale="Turbo",
                        opacity=0.85,
                        showscale=True,
                        colorbar=dict(title="Height (m)")
                    ),
                    name="LiDAR Point Cloud"
                ),

                go.Scatter3d(
                    x=box_x,
                    y=box_y,
                    z=box_z,
                    mode="lines",
                    line=dict(color="#FF3030", width=5),
                    name="Ground-Truth Vehicle Boxes"
                ),

                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        size=EGO_MARKER_SIZE,
                        color=EGO_MARKER_COLOR,
                        symbol="triangle-up",
                        line=dict(color="#00FF99", width=2)
                    ),
                    name="Ego Vehicle"
                ),

                go.Scatter(
                    x=world_points[:, 0],
                    y=world_points[:, 1],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=world_points[:, 2],
                        colorscale="Turbo",
                        opacity=0.8,
                        showscale=False
                    ),
                    name="BEV LiDAR Points"
                ),

                go.Scatter(
                    x=box_x,
                    y=box_y,
                    mode="lines",
                    line=dict(color="#FF3030", width=2),
                    name="BEV Ground-Truth Boxes"
                ),

                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        size=EGO_MARKER_SIZE,
                        color=EGO_MARKER_COLOR,
                        symbol="triangle-up",
                        line=dict(color="#00FF99", width=2)
                    ),
                    name="Ego Vehicle"
                ),
            ]
        )
    )


# -------------------------------
# INITIAL FRAME
# -------------------------------

initial_data = animation_frames[0].data

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    column_widths=[0.62, 0.38],
    subplot_titles=("3D LiDAR PERCEPTION", "BIRD'S-EYE VIEW (BEV)"),
    horizontal_spacing=0.06
)

# Add the four visualization traces to their respective panels
for trace in initial_data:
    if isinstance(trace, go.Scatter3d):
        fig.add_trace(trace, row=1, col=1)
    else:
        fig.add_trace(trace, row=1, col=2)

# Attach the existing animation frames
fig.frames = animation_frames

# -------------------------------
# PLAYBACK CONTROLS
# -------------------------------

slider_steps = []

for frame_index in range(FRAME_COUNT):

    slider_steps.append(
        dict(
            method="animate",
            args=[
                [str(frame_index)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=str(frame_index)
        )
    )

fig.update_layout(

    title=dict(
        text="WAYMO OPEN DATASET | 3D LIDAR PERCEPTION",
        x=0.5,
        font=dict(size=23)
    ),

    template="plotly_dark",

    scene=dict(
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        zaxis_title="Z Position (m)",
        aspectmode="data",
        bgcolor="#080B12"
    ),
    xaxis=dict(
        title="X Position (m)",
        showgrid=True,
        zeroline=False,
        scaleanchor="y",
        scaleratio=1
    ),

    yaxis=dict(
        title="Y Position (m)",
        showgrid=True,
        zeroline=False
    ),
    paper_bgcolor="#080B12",

    height=850,

    legend=dict(
        orientation="h",
        y=1.02,
        yanchor="bottom",
        x=0.5,
        xanchor="center",
        bgcolor="rgba(8, 11, 18, 0.8)"
    ),

    margin=dict(
        l=50,
        r=150,
        t=110,
        b=100
    ),

    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.05,
            y=1.12,
            buttons=[
                dict(
                    label="▶ PLAY",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                ),
                dict(
                    label="⏸ PAUSE",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0)
                        )
                    ]
                )
            ]
        )
    ],

    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Frame: "),
            steps=slider_steps,
            x=0.1,
            y=0,
            len=0.85
        )
    ],

    annotations=[
        dict(
            text="Red boxes = Waymo ground-truth annotations, not model predictions",
            x=0.5,
            y=-0.05,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#AAAAAA")
        )
    ]
)

# -------------------------------
# SAVE AND OPEN
# -------------------------------

output_file = OUTPUT_DIR / "waymo_animated_3d_viewer.html"

fig.write_html(
    str(output_file),
    include_plotlyjs=True,
    auto_open=True
)

print("\nANIMATION COMPLETED")
print("Frames:", FRAME_COUNT)
print("Saved:", output_file.resolve())