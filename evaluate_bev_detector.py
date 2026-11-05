
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn as nn
import random

# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
CHECKPOINT_PATH = Path("outputs/bev_detector/bev_detector_best.pt")
OUTPUT_DIR = Path("outputs/bev_detector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_THRESHOLD = 0.70
IOU_THRESHOLDS = (0.25, 0.50)
MAX_DETECTIONS_PER_FRAME = 50

# These settings must match train_bev_detector.py.
X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -40.0, 40.0
GRID_SIZE = 0.5

GRID_H = int((X_MAX - X_MIN) / GRID_SIZE)
GRID_W = int((Y_MAX - Y_MIN) / GRID_SIZE)

CLASS_NAMES = {"Car": 0, "Van": 1, "Truck": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_NAMES.items()}
NUM_CLASSES = len(CLASS_NAMES)
OUTPUT_CHANNELS = 12

NUM_POINTS = 16384

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# Must match the model architecture in train_bev_detector.py.
# ============================================================
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
            64,
            OUTPUT_CHANNELS,
            kernel_size=1,
        )

    def forward(self, x):
        return self.head(self.backbone(x))

def make_bev(raw_points):
    """
    Convert a LiDAR point cloud into a two-channel BEV grid.

    Channel 0: normalized maximum point height.
    Channel 1: logarithmically normalized point density.

    This preprocessing matches train_bev_detector.py.
    """
    points = raw_points[:, :4].astype(np.float32)

    valid = (
        (points[:, 0] >= X_MIN)
        & (points[:, 0] < X_MAX)
        & (points[:, 1] >= Y_MIN)
        & (points[:, 1] < Y_MAX)
        & np.isfinite(points[:, :3]).all(axis=1)
    )

    points = points[valid]

    bev = np.zeros(
        (2, GRID_H, GRID_W),
        dtype=np.float32,
    )

    if len(points) == 0:
        return bev

    ix = (
        (points[:, 0] - X_MIN) / GRID_SIZE
    ).astype(np.int32)

    iy = (
        (points[:, 1] - Y_MIN) / GRID_SIZE
    ).astype(np.int32)

    # Channel 0: normalized maximum height.
    normalized_heights = np.clip(
        (points[:, 2] + 3.0) / 6.0,
        0.0,
        1.0,
    )

    np.maximum.at(
        bev[0],
        (ix, iy),
        normalized_heights,
    )

    # Channel 1: logarithmically normalized point density.
    np.add.at(
        bev[1],
        (ix, iy),
        1.0,
    )

    bev[1] = np.clip(
        np.log1p(bev[1]) / np.log(65.0),
        0.0,
        1.0,
    )

    return bev





# ============================================================
# MANIFEST AND CALIBRATION
# ============================================================

def load_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("records", data.get("frames", []))
    else:
        raise ValueError("Unsupported manifest format.")

    if not isinstance(records, list):
        raise ValueError("Manifest records must be a list.")

    return records


def read_calibration(path):
    values = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, raw = line.split(":", 1)

            try:
                values[key.strip()] = np.asarray(
                    [float(v) for v in raw.split()],
                    dtype=np.float64,
                )
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing calibration entries in {path}")

    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)

    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)

    return rect @ velo_to_cam


# ============================================================
# GROUND-TRUTH BOX CONVERSION
# ============================================================

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
    yaw_camera = float(yaw_camera)

    if min(height, width, length) <= 0:
        return None

    center_camera = np.array(
        [x, y, z, 1.0],
        dtype=np.float64,
    )
    center_lidar = (camera_to_lidar @ center_camera)[:3]

    # KITTI camera-frame forward direction.
    forward_camera = np.array(
        [
            math.cos(yaw_camera),
            0.0,
            -math.sin(yaw_camera),
        ],
        dtype=np.float64,
    )

    rotation_camera_to_lidar = camera_to_lidar[:3, :3]
    forward_lidar = rotation_camera_to_lidar @ forward_camera

    yaw_lidar = math.atan2(
        forward_lidar[1],
        forward_lidar[0],
    )

    # KITTI location is the bottom center.
    # Convert to the geometric center in LiDAR coordinates.
    center_lidar[2] += height / 2.0

    return {
        "class_id": int(annotation.get("class_id", -1)),
        "center": center_lidar,
        "length": length,
        "width": width,
        "height": height,
        "yaw": yaw_lidar,
    }


def get_ground_truth_boxes(record, camera_to_lidar):
    annotations = record.get("annotations", [])
    boxes = []

    for annotation in annotations:
        class_name = annotation.get(
            "class",
            annotation.get("type", ""),
        )

        if class_name not in CLASS_NAMES:
            continue

        normalized = dict(annotation)
        normalized["class_id"] = CLASS_NAMES[class_name]

        box = camera_box_to_lidar(normalized, camera_to_lidar)

        if box is None:
            continue

        x, y, _ = box["center"]

        if X_MIN <= x < X_MAX and Y_MIN <= y < Y_MAX:
            boxes.append(box)

    return boxes


# ============================================================
# POINT CLOUD -> BEV
# Two channels: maximum height and point density.
# ============================================================



# ============================================================
# BOX DECODING
# Output channels:
# 0 objectness
# 1:4 class logits
# 4 fractional X
# 5 fractional Y
# 6 center Z
# 7 log length
# 8 log width
# 9 log height
# 10 sin(yaw)
# 11 cos(yaw)
# ============================================================

def decode_predictions(prediction, score_threshold=SCORE_THRESHOLD):
    """
    Decode BEV predictions using the 2-meter output grid
    defined by train_bev_detector.py.
    """
    prediction = prediction[0].detach().cpu().numpy()

    # Objectness probability.
    objectness = 1.0 / (
        1.0 + np.exp(-np.clip(prediction[0], -50, 50))
    )

    # Class probabilities.
    class_logits = prediction[1:4]
    class_logits = (
        class_logits
        - class_logits.max(axis=0, keepdims=True)
    )

    class_probs = np.exp(class_logits)
    class_probs /= class_probs.sum(axis=0, keepdims=True)

    class_ids = class_probs.argmax(axis=0)
    class_scores = class_probs.max(axis=0)

    # Combined confidence score.
    scores = objectness * class_scores

    # Training output cells are 2 meters wide.
    output_cell_size = GRID_SIZE * 4.0

    detections = []

    # First spatial index represents X; second represents Y.
    for ix, iy in np.argwhere(scores >= score_threshold):
        score = float(scores[ix, iy])
        class_id = int(class_ids[ix, iy])

        if class_id not in ID_TO_CLASS:
            continue

        # Decode fractional offsets within each 2-meter cell.
        frac_x = float(prediction[4, ix, iy])
        frac_y = float(prediction[5, ix, iy])

        center_x = X_MIN + (ix + frac_x) * output_cell_size
        center_y = Y_MIN + (iy + frac_y) * output_cell_size

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

        sin_yaw = float(prediction[10, ix, iy])
        cos_yaw = float(prediction[11, ix, iy])
        yaw = math.atan2(sin_yaw, cos_yaw)

        # Discard centers outside the detection region.
        if not (
            X_MIN <= center_x < X_MAX
            and Y_MIN <= center_y < Y_MAX
        ):
            continue

        detections.append({
            "class_id": class_id,
            "score": score,
            "center": np.array(
                [center_x, center_y, center_z],
                dtype=np.float64,
            ),
            "length": length,
            "width": width,
            "height": height,
            "yaw": yaw,
        })

    # Highest-confidence predictions first.
    detections.sort(
        key=lambda box: box["score"],
        reverse=True,
    )

    # Class-wise rotated BEV NMS.
    kept = []

    for box in detections:
        overlaps_existing = any(
            box["class_id"] == other["class_id"]
            and bev_iou(box, other) >= 0.15
            for other in kept
        )

        if not overlaps_existing:
            kept.append(box)

        if len(kept) >= MAX_DETECTIONS_PER_FRAME:
            break

    return kept

# ============================================================
# BEV IoU
# Rotated rectangle clipping uses a polygon intersection.
# ============================================================

def rectangle_corners(box):
    length = box["length"]
    width = box["width"]
    cx, cy = box["center"][:2]
    yaw = box["yaw"]

    # Counterclockwise corner order.
    local = np.array([
        [-length / 2, -width / 2],
        [ length / 2, -width / 2],
        [ length / 2,  width / 2],
        [-length / 2,  width / 2],
    ], dtype=np.float64)

    c = math.cos(yaw)
    s = math.sin(yaw)

    rotation = np.array([
        [c, -s],
        [s,  c]
    ])

    return local @ rotation.T + np.array([cx, cy])


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    ) * 0.5


def inside_edge(point, a, b):
    return (
        (b[0] - a[0]) * (point[1] - a[1])
        - (b[1] - a[1]) * (point[0] - a[0])
    ) >= -1e-9


def line_intersection(p1, p2, a, b):
    direction = p2 - p1
    edge = b - a

    denominator = (
        direction[0] * edge[1]
        - direction[1] * edge[0]
    )

    if abs(denominator) < 1e-12:
        return p2

    delta = a - p1
    t = (
        delta[0] * edge[1]
        - delta[1] * edge[0]
    ) / denominator

    return p1 + t * direction


def polygon_intersection(subject, clip):
    output = [point.copy() for point in subject]

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_list = output
        output = []

        if not input_list:
            break

        previous = input_list[-1]

        for current in input_list:
            current_inside = inside_edge(current, a, b)
            previous_inside = inside_edge(previous, a, b)

            if current_inside:
                if not previous_inside:
                    output.append(
                        line_intersection(previous, current, a, b)
                    )
                output.append(current)

            elif previous_inside:
                output.append(
                    line_intersection(previous, current, a, b)
                )

            previous = current

    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def bev_iou(box_a, box_b):
    poly_a = rectangle_corners(box_a)
    poly_b = rectangle_corners(box_b)

    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection = polygon_intersection(poly_a, poly_b)
    intersection_area = polygon_area(intersection)

    union = area_a + area_b - intersection_area

    if union <= 0:
        return 0.0

    return float(intersection_area / union)


# ============================================================
# METRICS
# ============================================================

def evaluate_threshold(all_frames, iou_threshold):
    total_gt = sum(len(frame["gt"]) for frame in all_frames)
    total_predictions = sum(len(frame["pred"]) for frame in all_frames)

    if total_gt == 0:
        return {
            "iou_threshold": iou_threshold,
            "ground_truth_boxes": 0,
            "predictions": total_predictions,
            "true_positives": 0,
            "false_positives": total_predictions,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "average_precision": 0.0,
        }

    prediction_list = []

    for frame_index, frame in enumerate(all_frames):
        for pred in frame["pred"]:
            prediction_list.append(
                (pred["score"], frame_index, pred)
            )

    prediction_list.sort(key=lambda item: item[0], reverse=True)

    matched = [
        set() for _ in all_frames
    ]

    tp_flags = []
    fp_flags = []
    debug_count = 0


    debug_count = 0

    for score, frame_index, pred in prediction_list:
        gt_boxes = all_frames[frame_index]["gt"]

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched[frame_index]:
                continue

            # Calculate overlap before checking the class.
            overlap = bev_iou(pred, gt)

            # Print the first 10 comparisons for diagnosis.
            if debug_count < 10:
                print(
                    f"DEBUG: frame_index={frame_index}, "
                    f"pred_class={pred['class_id']}, "
                    f"gt_class={gt['class_id']}, "
                    f"pred_center={pred['center']}, "
                    f"gt_center={gt['center']}, "
                    f"IoU={overlap:.4f}"
                )
                debug_count += 1

            # Only same-class boxes can be matched.
            if gt["class_id"] != pred["class_id"]:
                continue

            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched[frame_index].add(best_gt_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp = np.cumsum(tp_flags, dtype=np.float64)
    fp = np.cumsum(fp_flags, dtype=np.float64)

    recalls = tp / total_gt
    precisions = tp / np.maximum(tp + fp, 1e-12)

    final_tp = int(sum(tp_flags))
    final_fp = int(sum(fp_flags))
    final_fn = total_gt - final_tp

    precision = final_tp / max(final_tp + final_fp, 1)
    recall = final_tp / max(total_gt, 1)

    # All-points interpolated precision-recall area.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changing = np.where(mrec[1:] != mrec[:-1])[0]

    average_precision = float(
        np.sum(
            (mrec[changing + 1] - mrec[changing])
            * mpre[changing + 1]
        )
    )

    return {
        "iou_threshold": iou_threshold,
        "ground_truth_boxes": total_gt,
        "predictions": total_predictions,
        "true_positives": final_tp,
        "false_positives": final_fp,
        "false_negatives": final_fn,
        "precision": float(precision),
        "recall": float(recall),
        "average_precision": average_precision,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("BEV DETECTOR EVALUATION")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Train the model first."
        )

        # Reproduce the dataset selection used during training.
    SEED = 42
    MAX_FRAMES = 500

    records = load_manifest(MANIFEST_PATH)

    if not records:
        raise ValueError("The manifest contains no records.")

    # Keep usable records, matching the training selection.
    valid_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        if not record.get("lidar_path"):
            continue

        if not record.get("calibration_path"):
            continue

        if not record.get("annotations"):
            continue

        valid_records.append(record)

    # Shuffle and select the same maximum number of frames.
    random.seed(SEED)
    random.shuffle(valid_records)
    valid_records = valid_records[:MAX_FRAMES]

    if len(valid_records) < 10:
        raise RuntimeError(
            f"Only {len(valid_records)} usable frames found."
        )

    # Reproduce the 80/20 training/validation split.
    split_index = max(1, int(0.8 * len(valid_records)))

    train_records = valid_records[:split_index]
    val_records = valid_records[split_index:]

    # Evaluate only the held-out validation subset.
    records = val_records

    print(f"Usable selected frames: {len(valid_records)}")
    print(f"Training frames in reproduced split: {len(train_records)}")
    print(f"Validation frames to evaluate: {len(records)}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BEVDetector().to(DEVICE)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint has no 'model_state_dict'. "
            "Inspect the checkpoint format before proceeding."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Manifest frames: {len(records)}")

    all_frames = []
    processed = 0
    skipped = 0

    with torch.no_grad():
        for index, record in enumerate(records):
            lidar_value = record.get("lidar_path")
            calib_value = record.get("calibration_path")

            if not lidar_value or not calib_value:
                skipped += 1
                continue

            lidar_path = Path(lidar_value)
            calib_path = Path(calib_value)

            if not lidar_path.is_file() or not calib_path.is_file():
                skipped += 1
                continue

            raw = np.fromfile(lidar_path, dtype=np.float32)

            if raw.size == 0 or raw.size % 4 != 0:
                skipped += 1
                continue

            raw = raw.reshape(-1, 4)

            lidar_to_camera = read_calibration(calib_path)
            camera_to_lidar = np.linalg.inv(lidar_to_camera)

            bev = make_bev(raw)
            input_tensor = torch.from_numpy(bev).unsqueeze(0).to(DEVICE)

            prediction = model(input_tensor)

            pred_boxes = decode_predictions(prediction)
            gt_boxes = get_ground_truth_boxes(record, camera_to_lidar)

            all_frames.append({
                "frame_id": record.get("frame_id", str(index)),
                "pred": pred_boxes,
                "gt": gt_boxes,
            })

            processed += 1

            if processed % 25 == 0:
                print(
                    f"Processed {processed} frames | "
                    f"predictions: {sum(len(f['pred']) for f in all_frames)} | "
                    f"ground truth: {sum(len(f['gt']) for f in all_frames)}"
                )

    if processed == 0:
        raise RuntimeError(
            "No frames were evaluated. Check the paths in "
            "outputs/kitti_vehicle_manifest.json."
        )

    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "manifest": str(MANIFEST_PATH),
        "device": str(DEVICE),
        "score_threshold": SCORE_THRESHOLD,
        "processed_frames": processed,
        "skipped_frames": skipped,
        "metrics": [
            evaluate_threshold(all_frames, threshold)
            for threshold in IOU_THRESHOLDS
        ],
        "warning": (
            "These are BEV metrics, not KITTI official 3D AP. "
            "Confirm model architecture, target encoding, and decoding "
            "match train_bev_detector.py before interpreting results."
        ),
    }

    output_path = OUTPUT_DIR / "evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nEvaluation results")
    print("-" * 65)

    for metric in results["metrics"]:
        print(f"BEV IoU threshold: {metric['iou_threshold']:.2f}")
        print(f"  Ground truth boxes: {metric['ground_truth_boxes']}")
        print(f"  Predictions:        {metric['predictions']}")
        print(f"  True positives:     {metric['true_positives']}")
        print(f"  False positives:    {metric['false_positives']}")
        print(f"  False negatives:    {metric['false_negatives']}")
        print(f"  Precision:          {metric['precision']:.4f}")
        print(f"  Recall:             {metric['recall']:.4f}")
        print(f"  Average precision:  {metric['average_precision']:.4f}")
        print()

    print(f"Processed frames: {processed}")
    print(f"Skipped frames:   {skipped}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
