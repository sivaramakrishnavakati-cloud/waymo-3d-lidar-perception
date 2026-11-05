
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
CHECKPOINT_PATH = Path("outputs/bev_detector/bev_detector_latest.pt")
OUTPUT_DIR = Path("outputs/bev_visualizations")

# Choose a KITTI frame ID. Example: "000006".
# If that frame is not found, the script uses the first manifest record.
FRAME_ID = "000006"

# This is a visualization threshold, not a validated detection threshold.
CONFIDENCE_THRESHOLD = 0.10
MAX_DETECTIONS = 30

X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -40.0, 40.0
GRID_SIZE = 0.5

GRID_H = 140
GRID_W = 160
NUM_CLASSES = 3
OUTPUT_CHANNELS = 12

CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
}

# The model has two pooling layers, reducing 140x160 to 35x40.
OUTPUT_SCALE = 4
OUTPUT_CELL_SIZE = GRID_SIZE * OUTPUT_SCALE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL — MATCHES train_bev_detector.py
# ============================================================

class BEVDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(
            64,
            OUTPUT_CHANNELS,
            kernel_size=1,
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# DATA LOADING
# ============================================================

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run the dataset preparation steps first."
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        records = data.get("records", data.get("frames", []))
    else:
        records = data

    if not records:
        raise ValueError("No records found in the KITTI manifest.")

    return records


def get_frame_id(record):
    for key in ("frame_id", "image_id", "id", "sample_id"):
        if key in record:
            return str(record[key]).zfill(6)

    lidar_path = record.get("lidar_path", "")
    if lidar_path:
        return Path(lidar_path).stem

    return "unknown"


def resolve_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def load_lidar_points(lidar_path):
    path = resolve_path(lidar_path)

    if not path.exists():
        raise FileNotFoundError(f"LiDAR file not found: {path}")

    points = np.fromfile(path, dtype=np.float32)

    if points.size % 4 != 0:
        raise ValueError(
            f"Unexpected LiDAR file format: {path}"
        )

    return points.reshape(-1, 4)


# ============================================================
# BEV REPRESENTATION — MATCHES TRAINING ENCODING
# ============================================================

def make_bev(points):
    bev = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= X_MIN)
        & (x < X_MAX)
        & (y >= Y_MIN)
        & (y < Y_MAX)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(x) == 0:
        return bev

    ix = ((x - X_MIN) / GRID_SIZE).astype(np.int32)
    iy = ((y - Y_MIN) / GRID_SIZE).astype(np.int32)

    # Channel 0: maximum normalized point height.
    normalized_height = np.clip((z + 3.0) / 6.0, 0.0, 1.0)

    np.maximum.at(
        bev[0],
        (ix, iy),
        normalized_height,
    )

    # Channel 1: normalized log point count.
    counts = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    np.add.at(counts, (ix, iy), 1)

    bev[1] = np.clip(
        np.log1p(counts) / np.log(65.0),
        0.0,
        1.0,
    )

    return bev


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure training completed successfully."
        )

    model = BEVDetector().to(DEVICE)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


# ============================================================
# PREDICTION DECODING
# ============================================================

def decode_predictions(prediction):
    """
    Decode the model's 12 output channels.

    The corrected training script creates targets directly on
    the 35x40 output grid. Fractional X/Y offsets are relative to
    2-meter output cells, not 0.5-meter input cells.

    Predictions remain experimental and are not yet validated
    as accurate 3D bounding boxes.
    """

    objectness = torch.sigmoid(prediction[0]).cpu().numpy()
    class_logits = prediction[1:4].cpu().numpy()
    regression = prediction[4:12].cpu().numpy()

    # Convert class logits into the predicted class per cell.
    predicted_classes = np.argmax(class_logits, axis=0)

    candidates = np.argwhere(
        objectness >= CONFIDENCE_THRESHOLD
    )

    # Sort cells by objectness, highest first.
    candidates = sorted(
        candidates,
        key=lambda cell: objectness[cell[0], cell[1]],
        reverse=True,
    )

    detections = []

    for out_ix, out_iy in candidates:
        confidence = float(objectness[out_ix, out_iy])

        # Channels 4 and 5 contain fractional cell offsets.
        frac_x = float(regression[0, out_ix, out_iy])
        frac_y = float(regression[1, out_ix, out_iy])

        # Channels 6-8 contain center Z and log dimensions.
        center_z = float(regression[2, out_ix, out_iy])

        log_length = float(regression[3, out_ix, out_iy])
        log_width = float(regression[4, out_ix, out_iy])
        log_height = float(regression[5, out_ix, out_iy])

        # Channels 9 and 10 encode sine and cosine of yaw.
        yaw_sin = float(regression[6, out_ix, out_iy])
        yaw_cos = float(regression[7, out_ix, out_iy])

        # Each output cell covers a 2m x 2m region.
        # frac_x and frac_y are offsets within that output cell.
        center_x = X_MIN + (
            out_ix + frac_x
        ) * OUTPUT_CELL_SIZE

        center_y = Y_MIN + (
            out_iy + frac_y
        ) * OUTPUT_CELL_SIZE

        # Limit dimensions to prevent extreme predictions from
        # producing enormous polygons in the visualization.
        length = float(np.clip(np.exp(np.clip(log_length, -3, 5)), 0.5, 20.0))
        width = float(np.clip(np.exp(np.clip(log_width, -3, 4)), 0.3, 8.0))
        height = float(np.clip(np.exp(np.clip(log_height, -3, 3)), 0.3, 6.0))

        yaw = math.atan2(yaw_sin, yaw_cos)

        class_id = int(predicted_classes[out_ix, out_iy])

        detections.append({
            "confidence": confidence,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, "Unknown"),
            "x": center_x,
            "y": center_y,
            "z": center_z,
            "length": length,
            "width": width,
            "height": height,
            "yaw": yaw,
            "grid_x": int(out_ix),
            "grid_y": int(out_iy),
        })

        if len(detections) >= MAX_DETECTIONS:
            break

    return objectness, detections


# ============================================================
# VISUALIZATION
# ============================================================

def get_box_corners(detection):
    x = detection["x"]
    y = detection["y"]
    length = detection["length"]
    width = detection["width"]
    yaw = detection["yaw"]

    # Local box coordinates: length along X, width along Y.
    corners = np.array([
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
    ])

    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])

    rotated = corners @ rotation.T
    rotated[:, 0] += x
    rotated[:, 1] += y

    return rotated


def visualize(points, bev, objectness, detections, frame_id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same LiDAR range as the training BEV.
    valid = (
        (points[:, 0] >= X_MIN)
        & (points[:, 0] < X_MAX)
        & (points[:, 1] >= Y_MIN)
        & (points[:, 1] < Y_MAX)
        & np.isfinite(points[:, :3]).all(axis=1)
    )

    points = points[valid]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: LiDAR points viewed from above.
    axes[0].scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=0.25,
        cmap="viridis",
    )
    axes[0].set_xlim(X_MIN, X_MAX)
    axes[0].set_ylim(Y_MIN, Y_MAX)
    axes[0].set_xlabel("X — forward (m)")
    axes[0].set_ylabel("Y — left (m)")
    axes[0].set_title(f"LiDAR Top View — Frame {frame_id}")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)

    # Panel 2: Model objectness heatmap.
    im = axes[1].imshow(
        objectness.T,
        origin="lower",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=1,
    )
    axes[1].set_xlabel("X — forward (m)")
    axes[1].set_ylabel("Y — left (m)")
    axes[1].set_title("Predicted Objectness Heatmap (35 x 40 grid)")
    fig.colorbar(im, ax=axes[1], label="Objectness score")

    # Panel 3: Predicted BEV boxes overlaid on LiDAR points.
    axes[2].scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=0.25,
        cmap="viridis",
    )

    for index, detection in enumerate(detections):
        corners = get_box_corners(detection)

        polygon = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )

        axes[2].add_patch(polygon)

        axes[2].text(
            detection["x"],
            detection["y"],
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}",
            color="white",
            fontsize=7,
            bbox=dict(facecolor="red", alpha=0.65, pad=1),
        )

    axes[2].set_xlim(X_MIN, X_MAX)
    axes[2].set_ylim(Y_MIN, Y_MAX)
    axes[2].set_xlabel("X — forward (m)")
    axes[2].set_ylabel("Y — left (m)")
    axes[2].set_title(
        f"Experimental Predictions — {len(detections)} boxes"
    )
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"bev_predictions_{frame_id}.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BEV VEHICLE DETECTION — PREDICTION VISUALIZER")
    print("=" * 60)

    records = load_manifest()

    selected_record = next(
        (
            record
            for record in records
            if get_frame_id(record) == FRAME_ID
        ),
        None,
    )

    if selected_record is None:
        selected_record = records[0]
        print(
            f"Frame {FRAME_ID} was not found. "
            "Using the first manifest record instead."
        )

    frame_id = get_frame_id(selected_record)

    if "lidar_path" not in selected_record:
        raise KeyError(
            "The selected manifest record does not contain 'lidar_path'."
        )

    lidar_path = selected_record["lidar_path"]

    print(f"Frame: {frame_id}")
    print(f"LiDAR file: {lidar_path}")

    points = load_lidar_points(lidar_path)
    bev = make_bev(points)

    model = load_model()

    input_tensor = torch.from_numpy(bev).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor)[0]

    objectness, detections = decode_predictions(prediction)

    print(f"\nPoints loaded: {len(points):,}")
    print(
        "Highest objectness score: "
        f"{float(objectness.max()):.4f}"
    )
    print(f"Detections above threshold: {len(detections)}")
    print(f"Visualization threshold: {CONFIDENCE_THRESHOLD}")

    if detections:
        print("\nTop predicted detections:")

        for index, detection in enumerate(detections[:10], start=1):
            print(
                f"{index:02d}. {detection['class_name']:5s} "
                f"confidence={detection['confidence']:.3f} "
                f"center=({detection['x']:.2f}, "
                f"{detection['y']:.2f}, "
                f"{detection['z']:.2f}) "
                f"size=(L={detection['length']:.2f}, "
                f"W={detection['width']:.2f}, "
                f"H={detection['height']:.2f})"
            )

    visualize(
        points,
        bev,
        objectness,
        detections,
        frame_id,
    )


if __name__ == "__main__":
    main()

