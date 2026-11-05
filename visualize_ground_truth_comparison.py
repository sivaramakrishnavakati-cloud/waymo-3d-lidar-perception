
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# CONFIGURATION
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
CHECKPOINT_PATH = Path("outputs/bev_detector/bev_detector_latest.pt")
OUTPUT_DIR = Path("outputs/ground_truth_comparison")

FRAME_ID = "000001"
CONFIDENCE_THRESHOLD = 0.25
MAX_DETECTIONS = 30

X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -40.0, 40.0

GRID_SIZE = 0.5
GRID_H, GRID_W = 140, 160

OUTPUT_SCALE = 4
OUTPUT_CELL_SIZE = GRID_SIZE * OUTPUT_SCALE
OUTPUT_CHANNELS = 12

CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
}

CLASS_TO_ID = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# MODEL
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
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(
            64, OUTPUT_CHANNELS, kernel_size=1
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# MANIFEST AND FILE PATHS
def load_manifest():
    with MANIFEST_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get(
            "records", data.get("frames", [])
        )
    else:
        raise ValueError("Unsupported manifest format.")

    if not records:
        raise ValueError("No records found in manifest.")

    return records


def get_frame_id(record):
    for key in (
        "frame_id", "image_id", "id", "sample_id"
    ):
        if key in record:
            return str(record[key]).zfill(6)

    lidar_path = record.get("lidar_path", "")
    return Path(lidar_path).stem if lidar_path else "unknown"


def resolve_path(value):
    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def load_points(path_value):
    path = resolve_path(path_value)

    if not path.is_file():
        raise FileNotFoundError(
            f"LiDAR file not found: {path}"
        )

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR file: {path}")

    return raw.reshape(-1, 4)
# CALIBRATION AND BOX CONVERSION
def read_calibration(path):
    values = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, raw = line.split(":", 1)

            try:
                values[key.strip()] = np.array(
                    [float(v) for v in raw.split()],
                    dtype=np.float64,
                )
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing calibration values in {path}")

    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)

    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)

    return rect @ velo_to_cam


def camera_box_to_lidar(annotation, camera_to_lidar):
    dims = annotation.get(
        "dimensions_hwl",
        annotation.get("dimensions"),
    )
    location = annotation.get(
        "location_camera",
        annotation.get("location"),
    )
    yaw_camera = annotation.get(
        "rotation_y",
        annotation.get("yaw"),
    )

    if dims is None or location is None or yaw_camera is None:
        return None

    height, width, length = map(float, dims)
    x, y, z = map(float, location)

    if min(height, width, length) <= 0:
        return None

    center_camera = np.array(
        [x, y, z, 1.0],
        dtype=np.float64,
    )

    center_lidar = (camera_to_lidar @ center_camera)[:3]

    forward_camera = np.array(
        [
            math.cos(yaw_camera),
            0.0,
            -math.sin(yaw_camera),
        ],
        dtype=np.float64,
    )

    forward_lidar = (
        camera_to_lidar[:3, :3] @ forward_camera
    )

    yaw_lidar = math.atan2(
        forward_lidar[1],
        forward_lidar[0],
    )

    # KITTI camera annotations locate the box at its bottom.
    # Convert the center to the box's vertical midpoint.
    center_lidar[2] += height / 2.0

    cls = annotation.get(
        "class",
        annotation.get("type", ""),
    )

    return {
        "class_name": cls,
        "center": center_lidar.astype(np.float32),
        "length": length,
        "width": width,
        "height": height,
        "yaw": float(yaw_lidar),
    }


# INPUT PREPROCESSING
def make_bev(points):
    bev = np.zeros(
        (2, GRID_H, GRID_W),
        dtype=np.float32,
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

        # Channel 0: normalized maximum point height.
        np.maximum.at(
            bev[0],
            (ix, iy),
            np.clip(
                (pts[:, 2] + 3.0) / 6.0,
                0.0,
                1.0,
            ),
        )

        # Channel 1: normalized LiDAR point density.
        np.add.at(bev[1], (ix, iy), 1.0)

        bev[1] = np.clip(
            np.log1p(bev[1]) / np.log(65.0),
            0.0,
            1.0,
        )

    return bev


# MODEL LOADING
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
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


# PREDICTION DECODING
def decode_predictions(prediction):
    objectness = torch.sigmoid(
        prediction[0]
    ).cpu().numpy()

    print("\n--- CONFIDENCE DEBUG ---")
    print("Maximum objectness:", float(objectness.max()))
    print("Minimum objectness:", float(objectness.min()))
    print("Current threshold:", CONFIDENCE_THRESHOLD)
    print(
        "Cells above threshold:",
        int(np.sum(objectness >= CONFIDENCE_THRESHOLD))
    )
    print("------------------------\n")


    class_logits = prediction[1:4].cpu().numpy()
    regression = prediction[4:12].cpu().numpy()

    classes = np.argmax(class_logits, axis=0)

    candidates = np.argwhere(
        objectness >= CONFIDENCE_THRESHOLD
    )

    candidates = sorted(
        candidates,
        key=lambda p: objectness[p[0], p[1]],
        reverse=True,
    )

    detections = []

    for ix, iy in candidates[:MAX_DETECTIONS]:
        cx = X_MIN + (
            ix + float(regression[0, ix, iy])
        ) * OUTPUT_CELL_SIZE

        cy = Y_MIN + (
            iy + float(regression[1, ix, iy])
        ) * OUTPUT_CELL_SIZE

        yaw = math.atan2(
            float(regression[6, ix, iy]),
            float(regression[7, ix, iy]),
        )

        cls_id = int(classes[ix, iy])

        detections.append({
            "confidence": float(objectness[ix, iy]),
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(
                cls_id,
                "Unknown",
            ),
            "x": cx,
            "y": cy,
            "z": float(regression[2, ix, iy]),
            "length": float(np.clip(
                np.exp(np.clip(
                    regression[3, ix, iy], -3, 5
                )),
                0.5,
                20,
            )),
            "width": float(np.clip(
                np.exp(np.clip(
                    regression[4, ix, iy], -3, 4
                )),
                0.3,
                8,
            )),
            "height": float(np.clip(
                np.exp(np.clip(
                    regression[5, ix, iy], -3, 3
                )),
                0.3,
                6,
            )),
            "yaw": yaw,
        })

    return objectness, detections

# BOX GEOMETRY
def box_corners(x, y, length, width, yaw):
    half_l = length / 2.0
    half_w = width / 2.0

    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ], dtype=np.float32)

    c = math.cos(yaw)
    s = math.sin(yaw)

    rotation = np.array([
        [c, -s],
        [s, c],
    ], dtype=np.float32)

    return corners @ rotation.T + np.array([x, y])


def box_iou_2d(box_a, box_b):
    """
    Approximate BEV IoU.
    Uses rotated polygons when Shapely is installed;
    otherwise falls back to axis-aligned rectangles.
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon

        corners_a = box_corners(
            box_a["x"], box_a["y"],
            box_a["length"], box_a["width"],
            box_a["yaw"],
        )
        corners_b = box_corners(
            box_b["x"], box_b["y"],
            box_b["length"], box_b["width"],
            box_b["yaw"],
        )

        polygon_a = ShapelyPolygon(corners_a)
        polygon_b = ShapelyPolygon(corners_b)

        if not polygon_a.is_valid or not polygon_b.is_valid:
            return 0.0

        intersection = polygon_a.intersection(polygon_b).area
        union = polygon_a.area + polygon_b.area - intersection

        return float(intersection / union) if union > 0 else 0.0

    except ImportError:
        # Fallback: axis-aligned rectangles.
        ax1 = box_a["x"] - box_a["length"] / 2
        ax2 = box_a["x"] + box_a["length"] / 2
        ay1 = box_a["y"] - box_a["width"] / 2
        ay2 = box_a["y"] + box_a["width"] / 2

        bx1 = box_b["x"] - box_b["length"] / 2
        bx2 = box_b["x"] + box_b["length"] / 2
        by1 = box_b["y"] - box_b["width"] / 2
        by2 = box_b["y"] + box_b["width"] / 2

        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = iw * ih

        area_a = box_a["length"] * box_a["width"]
        area_b = box_b["length"] * box_b["width"]
        union = area_a + area_b - intersection

        return float(intersection / union) if union > 0 else 0.0


def ground_truth_to_detection(box):
    center = box["center"]

    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "length": float(box["length"]),
        "width": float(box["width"]),
        "height": float(box["height"]),
        "yaw": float(box["yaw"]),
        "class_name": box["class_name"],
        "class_id": CLASS_TO_ID.get(box["class_name"], -1),
    }


# NON-MAXIMUM SUPPRESSION (NMS)
def apply_nms(detections, iou_threshold=0.15):
    """
    Suppress overlapping predictions of the same class.

    Keeps the highest-confidence box and removes lower-confidence
    boxes whose BEV IoU exceeds the specified threshold.
    """
    if not detections:
        return []

    # Process each class separately.
    kept_detections = []

    class_ids = sorted({
        detection["class_id"]
        for detection in detections
    })

    for class_id in class_ids:
        class_detections = [
            detection
            for detection in detections
            if detection["class_id"] == class_id
        ]

        # Highest-confidence prediction first.
        class_detections.sort(
            key=lambda detection: detection["confidence"],
            reverse=True,
        )

        while class_detections:
            best = class_detections.pop(0)
            kept_detections.append(best)

            remaining = []

            for candidate in class_detections:
                overlap = box_iou_2d(best, candidate)

                if overlap <= iou_threshold:
                    remaining.append(candidate)

            class_detections = remaining

    # Return detections ordered by confidence.
    kept_detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    return kept_detections

# MATCHING
def match_predictions(ground_truth, predictions, iou_threshold=0.25):
    """
    Greedy matching by highest IoU.
    Matches are based on BEV overlap and class label.
    """
    possible_matches = []

    for gt_idx, gt in enumerate(ground_truth):
        gt_det = ground_truth_to_detection(gt)

        for pred_idx, pred in enumerate(predictions):
            if gt_det["class_id"] != pred["class_id"]:
                continue

            iou = box_iou_2d(gt_det, pred)

            if iou >= iou_threshold:
                possible_matches.append(
                    (iou, gt_idx, pred_idx)
                )

    possible_matches.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gt_idx, pred_idx in possible_matches:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    unmatched_gt = [
        i for i in range(len(ground_truth))
        if i not in matched_gt
    ]

    unmatched_pred = [
        i for i in range(len(predictions))
        if i not in matched_pred
    ]

    return matches, unmatched_gt, unmatched_pred


# VISUALIZATION
def draw_box(ax, box, color, label=None, linewidth=2):
    corners = box_corners(
        box["x"],
        box["y"],
        box["length"],
        box["width"],
        box["yaw"],
    )

    # The BEV image displays Y horizontally and X vertically.
    corners = corners[:, ::-1]

    polygon = Polygon(
        corners,
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )

    ax.add_patch(polygon)

    # Draw orientation indicator.
    front_x = box["x"] + (
        box["length"] / 2.0
    ) * math.cos(box["yaw"])

    front_y = box["y"] + (
        box["length"] / 2.0
    ) * math.sin(box["yaw"])

    ax.plot(
        [box["y"], front_y],
        [box["x"], front_x],
        color=color,
        linewidth=linewidth,
    )

    if label:
        ax.text(
            box["y"],
            box["x"],
            label,
            color=color,
            fontsize=7,
            ha="center",
            va="bottom",
        )


def plot_comparison(bev, objectness, ground_truth, predictions,
                    matches, unmatched_gt, unmatched_pred):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 8),
    )

    # Left: LiDAR BEV with boxes.
    axes[0].imshow(
        bev[1],
        origin="lower",
        extent=[Y_MIN, Y_MAX, X_MIN, X_MAX],
        aspect="auto",
        cmap="gray",
    )

    axes[0].set_title(
        f"Frame {FRAME_ID}: Ground Truth vs Predictions"
    )
    axes[0].set_xlabel("Y position (m)")
    axes[0].set_ylabel("X position (m)")

    # Ground-truth boxes in green.
    for gt_idx, gt in enumerate(ground_truth):
        gt_det = ground_truth_to_detection(gt)
        draw_box(
            axes[0],
            gt_det,
            color="lime",
            label=f"GT {gt_idx}",
        )

    # Matched predictions in blue.
    for gt_idx, pred_idx, iou in matches:
        pred = predictions[pred_idx]
        draw_box(
            axes[0],
            pred,
            color="deepskyblue",
            label=f"P {pred_idx} IoU={iou:.2f}",
        )

    # Unmatched predictions in red.
    for pred_idx in unmatched_pred:
        pred = predictions[pred_idx]
        draw_box(
            axes[0],
            pred,
            color="red",
            label=f"P {pred_idx} {pred['confidence']:.2f}",
        )

    axes[0].set_xlim(Y_MIN, Y_MAX)
    axes[0].set_ylim(X_MIN, X_MAX)
    axes[0].grid(alpha=0.15)

    # Right: model objectness heatmap.
    heatmap = axes[1].imshow(
        objectness,
        origin="lower",
        extent=[Y_MIN, Y_MAX, X_MIN, X_MAX],
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("Predicted Objectness")
    axes[1].set_xlabel("Y position (m)")
    axes[1].set_ylabel("X position (m)")

    fig.colorbar(
        heatmap,
        ax=axes[1],
        label="Objectness probability",
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{FRAME_ID}_comparison.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")


# MAIN
def main():
    print("Loading dataset manifest...")

    records = load_manifest()

    record = None

    for item in records:
        if get_frame_id(item) == FRAME_ID:
            record = item
            break

    if record is None:
        print(f"Frame {FRAME_ID} was not found in the manifest.")
        print("Available frame IDs include:")

        for item in records[:10]:
            print(" ", get_frame_id(item))

        return

    lidar_value = record.get(
        "lidar_path",
        record.get("velodyne_path"),
    )

    calibration_value = record.get(
        "calib_path",
        record.get("calibration_path"),
    )

    label_value = record.get(
        "label_path",
        record.get("labels_path"),
    )

    if not lidar_value:
        raise KeyError(
            "No LiDAR path found in the selected manifest record."
        )

    if not calibration_value:
        raise KeyError(
            "No calibration path found in the selected manifest record."
        )

    if not label_value:
        raise KeyError(
            "No label path found in the selected manifest record."
        )

    lidar_path = resolve_path(lidar_value)
    calibration_path = resolve_path(calibration_value)
    label_path = resolve_path(label_value)

    print(f"Frame ID: {FRAME_ID}")
    print(f"LiDAR file: {lidar_path}")
    print(f"Calibration file: {calibration_path}")
    print(f"Label file: {label_path}")

    points = load_points(lidar_path)
    bev = make_bev(points)

    camera_to_lidar = np.linalg.inv(
        read_calibration(calibration_path)
    )

    # Read standard KITTI label format (one object per line).
    annotations = []

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            # Standard KITTI labels contain 15 fields or more.
            if len(parts) < 15:
                continue

            try:
                annotation = {
                    "class": parts[0],
                    "truncation": float(parts[1]),
                    "occlusion": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bbox": [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    ],
                    # KITTI camera dimensions are height, width, length.
                    "dimensions_hwl": [
                        float(parts[8]),
                        float(parts[9]),
                        float(parts[10]),
                    ],
                    "location_camera": [
                        float(parts[11]),
                        float(parts[12]),
                        float(parts[13]),
                    ],
                    "rotation_y": float(parts[14]),
                }

                annotations.append(annotation)

            except ValueError:
                 print(f"Skipping invalid label line: {line.strip()}")

    print(f"Loaded {len(annotations)} label annotations.")

    ground_truth = []

    for annotation in annotations:
        cls = annotation.get(
            "class",
            annotation.get("type", ""),
        )

        if cls not in CLASS_TO_ID:
            continue

        box = camera_box_to_lidar(
            annotation,
            camera_to_lidar,
        )

        if box is not None:
            ground_truth.append(box)

    model = load_model()

    input_tensor = torch.from_numpy(
        bev
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor)[0]

    
    objectness, predictions = decode_predictions(
        prediction
    )

    print(f"Predictions before NMS: {len(predictions)}")

    predictions = apply_nms(
        predictions,
        iou_threshold=0.15,
    )

    print(f"Predictions after NMS: {len(predictions)}")

    matches, unmatched_gt, unmatched_pred = match_predictions(
        ground_truth,
        predictions,
    )



    true_positives = len(matches)
    false_positives = len(unmatched_pred)
    false_negatives = len(unmatched_gt)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    print("\n" + "=" * 50)
    print(f"Frame: {FRAME_ID}")
    print(f"Ground-truth vehicles: {len(ground_truth)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Matched predictions: {true_positives}")
    print(f"Unmatched predictions: {false_positives}")
    print(f"Missed ground-truth boxes: {false_negatives}")
    print(f"Single-frame precision: {precision:.3f}")
    print(f"Single-frame recall: {recall:.3f}")
    print("=" * 50)

    plot_comparison(
        bev,
        objectness,
        ground_truth,
        predictions,
        matches,
        unmatched_gt,
        unmatched_pred,
    )


if __name__ == "__main__":
    main()