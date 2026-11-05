import json
import math
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="3D LiDAR Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 3D LiDAR Vehicle Detection")
st.caption("Experimental BEV-based 3D vehicle detection using KITTI")

MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")

CHECKPOINT_PATH = Path(
    "outputs/bev_detector/bev_detector_latest.pt"
)

X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -40.0, 40.0

GRID_SIZE = 0.5
OUTPUT_SCALE = 4
OUTPUT_CELL_SIZE = GRID_SIZE * OUTPUT_SCALE

GRID_H = 140
GRID_W = 160

OUTPUT_CHANNELS = 12

CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# MODEL ARCHITECTURE
# Must match train_bev_detector.py
# ==========================================

class BEVDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Conv2d(
            64,
            OUTPUT_CHANNELS,
            kernel_size=1
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ==========================================
# LOAD KITTI DATASET
# ==========================================

@st.cache_data
def load_manifest():

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        records = data.get(
            "records",
            data.get("frames", [])
        )

    else:
        raise ValueError("Unsupported manifest format.")

    if not records:
        raise ValueError("No KITTI frames found.")

    return records


def get_frame_id(record, index):

    for key in ("frame_id", "image_id", "id", "sample_id"):

        if key in record:
            return str(record[key]).zfill(6)

    return str(index).zfill(6)


def resolve_path(value):

    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


# ==========================================
# LOAD LIDAR POINT CLOUD
# ==========================================

def load_lidar_points(lidar_path):

    path = resolve_path(lidar_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"LiDAR file not found: {path}"
        )

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError("Invalid KITTI LiDAR file.")

    return raw.reshape(-1, 4)


# ==========================================
# POINT CLOUD TO BEV
# ==========================================

def make_bev(points):

    bev = np.zeros(
        (2, GRID_H, GRID_W),
        dtype=np.float32
    )

    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (points[:, 0] >= X_MIN)
        & (points[:, 0] < X_MAX)
        & (points[:, 1] >= Y_MIN)
        & (points[:, 1] < Y_MAX)
    )

    pts = points[valid]

    if len(pts):

        ix = (
            (pts[:, 0] - X_MIN) / GRID_SIZE
        ).astype(np.int32)

        iy = (
            (pts[:, 1] - Y_MIN) / GRID_SIZE
        ).astype(np.int32)

        np.maximum.at(
            bev[0],
            (ix, iy),
            np.clip(
                (pts[:, 2] + 3.0) / 6.0,
                0.0,
                1.0
            )
        )

        np.add.at(
            bev[1],
            (ix, iy),
            1.0
        )

        bev[1] = np.clip(
            np.log1p(bev[1]) / np.log(65.0),
            0.0,
            1.0
        )

    return bev


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = BEVDetector().to(DEVICE)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state)

    model.eval()

    return model, checkpoint


# ==========================================
# RUN MODEL PREDICTION
# ==========================================

def predict_frame(model, points):

    bev = make_bev(points)

    input_tensor = torch.from_numpy(
        bev
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prediction = model(input_tensor)

    return prediction[0].detach().cpu().numpy()    

# ==========================================
# DECODE MODEL PREDICTIONS
# ==========================================

def decode_predictions(
    prediction,
    confidence_threshold=0.70,
    max_detections=30
):

    objectness = 1.0 / (
        1.0 + np.exp(
            -np.clip(prediction[0], -50, 50)
        )
    )

    class_logits = prediction[1:4]

    class_logits = (
        class_logits
        - class_logits.max(axis=0, keepdims=True)
    )

    class_probs = np.exp(class_logits)

    class_probs /= class_probs.sum(
        axis=0,
        keepdims=True
    )

    class_ids = class_probs.argmax(axis=0)
    class_scores = class_probs.max(axis=0)

    scores = objectness * class_scores

    detections = []

    candidates = np.argwhere(
        scores >= confidence_threshold
    )

    candidates = sorted(
        candidates,
        key=lambda cell: scores[cell[0], cell[1]],
        reverse=True
    )

    for ix, iy in candidates:

        if len(detections) >= max_detections:
            break

        class_id = int(class_ids[ix, iy])

        frac_x = float(prediction[4, ix, iy])
        frac_y = float(prediction[5, ix, iy])

        center_x = (
            X_MIN + (ix + frac_x) * OUTPUT_CELL_SIZE
        )

        center_y = (
            Y_MIN + (iy + frac_y) * OUTPUT_CELL_SIZE
        )

        center_z = float(prediction[6, ix, iy])

        length = math.exp(
            float(np.clip(prediction[7, ix, iy], -3, 5))
        )

        width = math.exp(
            float(np.clip(prediction[8, ix, iy], -3, 4))
        )

        height = math.exp(
            float(np.clip(prediction[9, ix, iy], -3, 3))
        )

        yaw = math.atan2(
            float(prediction[10, ix, iy]),
            float(prediction[11, ix, iy])
        )

        detections.append({
            "class": CLASS_NAMES[class_id],
            "confidence": float(scores[ix, iy]),
            "x": center_x,
            "y": center_y,
            "z": center_z,
            "length": length,
            "width": width,
            "height": height,
            "yaw": yaw
        })

    return detections


# ==========================================
# CREATE 3D BOX CORNERS
# ==========================================

def get_3d_box_corners(detection):

    x = detection["x"]
    y = detection["y"]
    z = detection["z"]

    length = detection["length"]
    width = detection["width"]
    height = detection["height"]

    yaw = detection["yaw"]

    local_corners = np.array([
        [-length / 2, -width / 2, -height / 2],
        [ length / 2, -width / 2, -height / 2],
        [ length / 2,  width / 2, -height / 2],
        [-length / 2,  width / 2, -height / 2],

        [-length / 2, -width / 2, height / 2],
        [ length / 2, -width / 2, height / 2],
        [ length / 2,  width / 2, height / 2],
        [-length / 2,  width / 2, height / 2]
    ])

    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])

    return local_corners @ rotation.T + np.array([x, y, z])


# ==========================================
# DRAW 3D VEHICLE BOXES
# ==========================================

def add_3d_boxes(fig, detections):

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for detection in detections:

        corners = get_3d_box_corners(detection)

        for start, end in edges:

            fig.add_trace(
                go.Scatter3d(
                    x=[
                        corners[start, 0],
                        corners[end, 0]
                    ],
                    y=[
                        corners[start, 1],
                        corners[end, 1]
                    ],
                    z=[
                        corners[start, 2],
                        corners[end, 2]
                    ],
                    mode="lines",
                    line=dict(
                        color="red",
                        width=5
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=[detection["x"]],
                y=[detection["y"]],
                z=[detection["z"]],
                mode="markers+text",
                marker=dict(
                    size=3,
                    color="red"
                ),
                text=[
                    f'{detection["class"]} '
                    f'{detection["confidence"]:.2f}'
                ],
                textposition="top center",
                name=detection["class"]
            )
        )

# ==========================================
# CREATE INTERACTIVE 3D VISUALIZATION
# ==========================================

def create_3d_visualization(points, detections):

    fig = go.Figure()

    # Limit displayed points for smoother rendering.
    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (points[:, 0] >= X_MIN)
        & (points[:, 0] < X_MAX)
        & (points[:, 1] >= Y_MIN)
        & (points[:, 1] < Y_MAX)
    )

    points = points[valid]

    if len(points) > 30000:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(points), 30000, replace=False
        )
        points = points[indices]

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=points[:, 3],
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Intensity")
            ),
            name="LiDAR Points"
        )
    )

    add_3d_boxes(fig, detections)

    fig.update_layout(
        title="3D LiDAR Vehicle Detection",
        scene=dict(
            xaxis_title="X - Forward (m)",
            yaxis_title="Y - Left (m)",
            zaxis_title="Z - Height (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2)
            )
        ),
        height=750,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


# ==========================================
# DASHBOARD APPLICATION
# ==========================================

def main():

    st.sidebar.header("Detection Settings")

    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05
    )

    max_detections = st.sidebar.slider(
        "Maximum Detections",
        min_value=5,
        max_value=50,
        value=30,
        step=5
    )

    st.sidebar.write(f"Device: {DEVICE}")

    try:
        records = load_manifest()
        model, checkpoint = load_model()

    except Exception as error:
        st.error(f"Loading error: {error}")
        st.stop()

    frame_options = [
        f"{get_frame_id(record, index)}"
        for index, record in enumerate(records)
    ]

    selected_frame = st.selectbox(
        "Select KITTI Frame",
        options=range(len(records)),
        format_func=lambda index: frame_options[index]
    )

    record = records[selected_frame]

    frame_id = get_frame_id(
        record, selected_frame
    )

    lidar_path = record.get("lidar_path")

    if not lidar_path:
        st.error("LiDAR path missing from manifest record.")
        st.stop()

    if st.button("Run 3D Detection", type="primary"):

        try:
            with st.spinner("Processing LiDAR frame..."):

                points = load_lidar_points(lidar_path)

                prediction = predict_frame(
                    model, points
                )

                detections = decode_predictions(
                    prediction,
                    confidence_threshold=confidence,
                    max_detections=max_detections
                )

            st.session_state["points"] = points
            st.session_state["detections"] = detections
            st.session_state["frame_id"] = frame_id

        except Exception as error:
            st.error(f"Detection error: {error}")
            st.stop()

    if "detections" in st.session_state:

        points = st.session_state["points"]
        detections = st.session_state["detections"]

        st.subheader(
            f"Results — Frame {st.session_state['frame_id']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "LiDAR Points",
            f"{len(points):,}"
        )

        col2.metric(
            "Detected Vehicles",
            len(detections)
        )

        col3.metric(
            "Device",
            str(DEVICE).upper()
        )

        st.plotly_chart(
            create_3d_visualization(
                points, detections
            ),
            use_container_width=True
        )

        st.subheader("Detection Details")

        if detections:

            st.dataframe(
                [
                    {
                        "Class": d["class"],
                        "Confidence": round(d["confidence"], 3),
                        "X (m)": round(d["x"], 2),
                        "Y (m)": round(d["y"], 2),
                        "Z (m)": round(d["z"], 2),
                        "Length": round(d["length"], 2),
                        "Width": round(d["width"], 2),
                        "Height": round(d["height"], 2)
                    }
                    for d in detections
                ],
                use_container_width=True
            )

        else:
            st.warning(
                "No detections at this confidence threshold. "
                "Try lowering the threshold."
            )


if __name__ == "__main__":
    main()