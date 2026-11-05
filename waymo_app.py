import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Waymo 3D Perception",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Automotive 3D Perception Dashboard")
st.caption("Waymo Open Dataset | LiDAR Perception Research")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    points = np.load(
        "data/waymo/processed/waymo_frame_000.npy"
    )

    labels = pd.read_csv(
        "data/waymo/processed/waymo_labels_000.csv"
    )

    return points, labels


# ---------------- MULTI-FRAME DATA ----------------

FRAME_DIR = Path("data/waymo/processed/multi_frame")

frame_files = sorted(FRAME_DIR.glob("frame_*.npy"))

if not frame_files:
    st.error("No Waymo frames found!")
    st.stop()

# Initialize frame selection
# ---------------- SEQUENCE PLAYBACK ----------------

if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0

if "playing" not in st.session_state:
    st.session_state.playing = False

def previous_frame():
    st.session_state.playing = False
    st.session_state.frame_index = max(
        0, st.session_state.frame_index - 1
    )

def next_frame():
    st.session_state.playing = False
    st.session_state.frame_index = min(
        len(frame_files) - 1,
        st.session_state.frame_index + 1
    )

st.subheader("Waymo Sequence Player")

col1, col2, col3, col4 = st.columns([1, 1, 3, 1])

with col1:
    st.button("◀ Previous", on_click=previous_frame)

with col2:
    if st.session_state.playing:
        if st.button("⏸ Pause"):
            st.session_state.playing = False
            st.rerun()
    else:
        if st.button("▶ Play"):
            st.session_state.playing = True
            st.rerun()

with col3:
    playback_speed = st.selectbox(
        "Playback Speed",
        options=[1, 2, 5],
        format_func=lambda x: f"{x}x",
        index=0
    )

with col4:
    st.button("Next ▶", on_click=next_frame)

# Automatic frame advancement
if st.session_state.playing:

    st.session_state.frame_index = (
        st.session_state.frame_index + 1
    ) % len(frame_files)

    st_autorefresh(
        interval=int(1000 / playback_speed),
        key="waymo_playback"
    )

# Frame slider
selected_frame = st.slider(
    "Frame Timeline",
    min_value=0,
    max_value=len(frame_files) - 1,
    key="frame_index"
)

# Load selected frame
frame_path = frame_files[selected_frame]
label_path = frame_path.with_suffix(".csv")

points = np.load(frame_path)
labels = pd.read_csv(label_path)

vehicles = labels[labels["type"] == 1]

st.caption(
    f"Frame {selected_frame + 1} / {len(frame_files)}"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Perception Controls")

point_limit = st.sidebar.slider(
    "LiDAR Points",
    min_value=5000,
    max_value=100000,
    value=30000,
    step=5000
)

show_boxes = st.sidebar.checkbox(
    "Show Ground-Truth Bounding Boxes",
    value=True
)

# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

col1.metric("Total LiDAR Points", f"{len(points):,}")
col2.metric("Ground-Truth Objects", len(labels))
col3.metric("Vehicles", len(vehicles))

st.divider()

# ---------------- 3D VIEWER ----------------

st.subheader("3D LiDAR Point Cloud")

rng = np.random.default_rng(42)

sample = points[
    rng.choice(
        len(points),
        min(point_limit, len(points)),
        replace=False
    )
]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=sample[:, 0],
    y=sample[:, 1],
    z=sample[:, 2],
    mode="markers",
    marker=dict(
        size=1.5,
        color=sample[:, 2],
        colorscale="Viridis",
        opacity=0.8,
        colorbar=dict(title="Height (m)")
    ),
    name="LiDAR Points"
))

# ---------------- GROUND-TRUTH BOXES ----------------

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

if show_boxes:

    for _, obj in vehicles.iterrows():

        length = obj["length"]
        width = obj["width"]
        height = obj["height"]
        heading = obj["heading"]

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

        rotation = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading),  np.cos(heading), 0],
            [0, 0, 1]
        ])

        corners = corners @ rotation.T

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
                line=dict(color="#FF3030", width=5),
                showlegend=False,
                hoverinfo="skip"
            ))

fig.update_layout(
    template="plotly_dark",
    height=750,
    margin=dict(l=0, r=0, t=30, b=0),
    scene=dict(
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        zaxis_title="Z (meters)",
        aspectmode="data"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- BIRD'S-EYE VIEW ----------------

st.divider()
st.subheader("Bird's-Eye View | BEV")

bev_fig = go.Figure()

# Plot LiDAR points from above
bev_fig.add_trace(go.Scattergl(
    x=sample[:, 0],
    y=sample[:, 1],
    mode="markers",
    marker=dict(
        size=2,
        color=sample[:, 2],
        colorscale="Viridis",
        opacity=0.7,
        showscale=True,
        colorbar=dict(title="Height (m)")
    ),
    name="LiDAR Points"
))

# Draw top-down ground-truth vehicle boxes
if show_boxes:

    for _, obj in vehicles.iterrows():

        length = obj["length"]
        width = obj["width"]
        heading = obj["heading"]

        corners = np.array([
            [-length/2, -width/2],
            [ length/2, -width/2],
            [ length/2,  width/2],
            [-length/2,  width/2],
            [-length/2, -width/2]
        ])

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)]
        ])

        corners = corners @ rotation.T

        corners[:, 0] += obj["center_x"]
        corners[:, 1] += obj["center_y"]

        bev_fig.add_trace(go.Scatter(
            x=corners[:, 0],
            y=corners[:, 1],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False
        ))

bev_fig.update_layout(
    template="plotly_dark",
    height=650,
    xaxis_title="X (meters)",
    yaxis_title="Y (meters)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(bev_fig, use_container_width=True)

# ---------------- OBJECT INSPECTOR ----------------

st.divider()

st.subheader("Vehicle Object Inspector")

if len(vehicles) > 0:

    vehicle_options = list(range(len(vehicles)))

    selected = st.selectbox(
        "Select Vehicle",
        vehicle_options
    )

    vehicle = vehicles.iloc[selected]

    a, b, c = st.columns(3)

    a.metric("Length", f"{vehicle['length']:.2f} m")
    b.metric("Width", f"{vehicle['width']:.2f} m")
    c.metric("Height", f"{vehicle['height']:.2f} m")

    st.write("Vehicle Position and Orientation")

    st.json({
        "Center X": round(float(vehicle["center_x"]), 3),
        "Center Y": round(float(vehicle["center_y"]), 3),
        "Center Z": round(float(vehicle["center_z"]), 3),
        "Heading": round(float(vehicle["heading"]), 3)
    })

st.caption(
    "Red boxes represent Waymo ground-truth annotations, "
    "not predictions from the trained BEV model."
)