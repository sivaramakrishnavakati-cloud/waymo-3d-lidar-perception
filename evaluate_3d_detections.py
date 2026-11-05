"""Evaluate cluster-based LiDAR detections against KITTI annotations.

This is a prototype evaluator. It compares axis-aligned 3D bounds in LiDAR
coordinates, not the official oriented KITTI 3D IoU metric.
Run from the project root: python evaluate_3d_detections.py
"""
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.cluster import DBSCAN

MODEL_PATH = Path("outputs/lidar_point_classifier_v2.pth")
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
OUTPUT_DIR = Path("outputs")
FRAME_ID = "000006"
MAX_POINTS = 100_000
THRESHOLD_OVERRIDE = None  # None uses checkpoint's recommended threshold
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 8
MIN_CLUSTER_POINTS = 20
SUPPORTED_CLASSES = {"Car", "Van", "Truck"}
IOU_THRESHOLDS = (0.25, 0.50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, points):
        return self.network(points).squeeze(-1)


def find_frame(manifest, frame_id):
    for record in manifest:
        lidar_path = Path(record.get("lidar_path", ""))
        record_id = str(record.get("frame_id", lidar_path.stem)).zfill(6)
        if (record_id == frame_id or lidar_path.stem == frame_id) and record.get("annotations") and lidar_path.is_file():
            return record
    raise RuntimeError(f"Annotated frame {frame_id} was not found in {MANIFEST_PATH}.")


def read_calibration(path):
    values = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = np.asarray([float(v) for v in raw.split()], dtype=np.float64)
            except ValueError:
                continue
    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Calibration file is missing Tr_velo_to_cam or R0_rect: {path}")
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    return rect @ velo_to_cam


def ground_truth_bounds_lidar(annotation, camera_to_lidar):
    """Convert KITTI camera box corners to LiDAR coordinates, then AABB bounds."""
    h, w, length = map(float, annotation["dimensions_hwl"])
    x, y, z = map(float, annotation["location_camera"])
    yaw = float(annotation["rotation_y"])
    local = np.array([
        [length / 2, length / 2, -length / 2, -length / 2,
         length / 2, length / 2, -length / 2, -length / 2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [w / 2, -w / 2, -w / 2, w / 2,
         w / 2, -w / 2, -w / 2, w / 2],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    corners_camera = (rot @ local + np.array([[x], [y], [z]])).T
    corners_h = np.column_stack((corners_camera, np.ones(len(corners_camera))))
    corners_lidar = (camera_to_lidar @ corners_h.T).T[:, :3]
    return corners_lidar.min(axis=0), corners_lidar.max(axis=0)


def generate_predictions(record, checkpoint, model):
    raw = np.fromfile(record["lidar_path"], dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Invalid LiDAR file: {record['lidar_path']}")
    all_points = raw.reshape(-1, 4)
    rng = np.random.default_rng(42)
    if len(all_points) > MAX_POINTS:
        indices = rng.choice(len(all_points), MAX_POINTS, replace=False)
        sampled = all_points[indices].copy()
    else:
        sampled = all_points.copy()

    xyz = sampled[:, :3].astype(np.float64)
    features = sampled.copy()
    features[:, :3] /= 50.0  # Must match the existing training/prediction pipeline.
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    threshold = (float(checkpoint.get("recommended_threshold", 0.50))
                 if THRESHOLD_OVERRIDE is None else float(THRESHOLD_OVERRIDE))
    vehicle_points = xyz[probs >= threshold]

    detections = []
    if len(vehicle_points) >= DBSCAN_MIN_SAMPLES:
        cluster_ids = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit_predict(vehicle_points)
        for cluster_id in sorted(set(cluster_ids) - {-1}):
            cluster = vehicle_points[cluster_ids == cluster_id]
            if len(cluster) < MIN_CLUSTER_POINTS:
                continue
            low = np.percentile(cluster, 2, axis=0)
            high = np.percentile(cluster, 98, axis=0)
            size = high - low
            if not (1.0 <= size[0] <= 18.0 and 0.8 <= size[1] <= 5.0 and 0.4 <= size[2] <= 5.0):
                continue
            detections.append({"low": low, "high": high, "points": len(cluster)})
    detections.sort(key=lambda d: d["points"], reverse=True)
    return detections, threshold, len(all_points), len(sampled), int(len(vehicle_points))


def iou_3d_aabb(a, b):
    low_a, high_a = a
    low_b, high_b = b
    intersection_size = np.maximum(0.0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b))
    intersection = float(np.prod(intersection_size))
    volume_a = float(np.prod(np.maximum(0.0, high_a - low_a)))
    volume_b = float(np.prod(np.maximum(0.0, high_b - low_b)))
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_matches(detections, ground_truth, threshold):
    pairs = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(ground_truth):
            iou = iou_3d_aabb((det["low"], det["high"]), gt)
            if iou >= threshold:
                pairs.append((iou, di, gi))
    pairs.sort(reverse=True)
    used_dets, used_gts, matched_ious = set(), set(), []
    for iou, di, gi in pairs:
        if di not in used_dets and gi not in used_gts:
            used_dets.add(di)
            used_gts.add(gi)
            matched_ious.append(iou)
    tp = len(matched_ious)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0}


def main():
    for path in (MODEL_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = find_frame(manifest, FRAME_ID)

    calibration_path_value = record.get("calibration_path")
    if not calibration_path_value:
        raise KeyError("Manifest record has no 'calibration_path' field; inspect prepare_kitti_dataset.py.")
    calibration_path = Path(calibration_path_value)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path.resolve()}")
    lidar_to_camera = read_calibration(calibration_path)
    camera_to_lidar = np.linalg.inv(lidar_to_camera)

    annotations = [a for a in record["annotations"] if a.get("class") in SUPPORTED_CLASSES]
    ground_truth = [ground_truth_bounds_lidar(a, camera_to_lidar) for a in annotations]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if checkpoint.get("feature_count") != 4:
        raise ValueError("Checkpoint does not specify four input features.")
    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    detections, threshold, original_count, sampled_count, predicted_points = generate_predictions(record, checkpoint, model)
    print("=" * 64)
    print("PROTOTYPE 3D DETECTION EVALUATION — SINGLE KITTI FRAME")
    print("=" * 64)
    print(f"Frame: {FRAME_ID} | Device: {DEVICE}")
    print(f"LiDAR points: {original_count:,} original; {sampled_count:,} evaluated")
    print(f"Point threshold: {threshold:.2f} | predicted vehicle points: {predicted_points:,}")
    print(f"Ground-truth vehicles: {len(ground_truth)} | candidate boxes: {len(detections)}")
    print("\nGround-truth labels:")
    for i, annotation in enumerate(annotations, 1):
        print(f"  GT {i}: {annotation['class']}")

    rows = []
    for threshold_iou in IOU_THRESHOLDS:
        result = greedy_matches(detections, ground_truth, threshold_iou)
        rows.append({"frame_id": FRAME_ID, "iou_threshold": threshold_iou, **result,
                     "candidate_boxes": len(detections), "ground_truth_boxes": len(ground_truth),
                     "point_threshold": threshold})
        print(f"\nAxis-aligned 3D IoU threshold: {threshold_iou:.2f}")
        print(f"  TP={result['tp']} | FP={result['fp']} | FN={result['fn']}")
        print(f"  Precision={result['precision']:.3f} | Recall={result['recall']:.3f} | F1={result['f1']:.3f}")
        print(f"  Mean IoU of matched boxes={result['mean_matched_iou']:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"3d_detection_evaluation_{FRAME_ID}.json"
    report_path.write_text(json.dumps({
        "note": "Prototype metric: axis-aligned 3D IoU in LiDAR coordinates; not official KITTI oriented 3D IoU.",
        "frame_id": FRAME_ID,
        "point_threshold": threshold,
        "candidate_boxes": len(detections),
        "ground_truth_boxes": len(ground_truth),
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")
    print("Note: this is a single-frame diagnostic. Axis-aligned IoU is an approximation, not official KITTI evaluation.")


if __name__ == "__main__":
    main()
