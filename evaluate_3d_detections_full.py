"""Full-dataset evaluation for the prototype KITTI LiDAR vehicle detector.

Run from the project root:
    python evaluate_3d_detections_full.py

This evaluates frames with supported vehicle annotations using the existing
point classifier + DBSCAN cluster proposal pipeline. It uses axis-aligned 3D
IoU in LiDAR coordinates, which is a diagnostic approximation, not official
KITTI oriented 3D detection evaluation.
"""
from pathlib import Path
import json
import time

import numpy as np
import torch
from sklearn.cluster import DBSCAN

MODEL_PATH = Path("outputs/lidar_point_classifier_v2.pth")
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
OUTPUT_DIR = Path("outputs")

# Start with all annotated frames. Set MAX_FRAMES = 100 for a quick smoke test.
MAX_FRAMES = None
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


def record_frame_id(record):
    lidar_path = Path(record.get("lidar_path", ""))
    return str(record.get("frame_id", lidar_path.stem)).zfill(6)


def read_calibration(path):
    values = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = np.asarray(
                    [float(v) for v in raw.split()], dtype=np.float64
                )
            except ValueError:
                continue
    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Calibration missing Tr_velo_to_cam or R0_rect: {path}")
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    return rect @ velo_to_cam


def ground_truth_bounds_lidar(annotation, camera_to_lidar):
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


def iou_3d_aabb(a, b):
    low_a, high_a = a
    low_b, high_b = b
    intersection_size = np.maximum(
        0.0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
    )
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_ious": matched_ious,
    }


def predict_detections(record, checkpoint, model):
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
    features[:, :3] /= 50.0  # Match the existing training/prediction pipeline.
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float().to(DEVICE))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    threshold = (
        float(checkpoint.get("recommended_threshold", 0.50))
        if THRESHOLD_OVERRIDE is None else float(THRESHOLD_OVERRIDE)
    )
    vehicle_points = xyz[probabilities >= threshold]
    detections = []

    if len(vehicle_points) >= DBSCAN_MIN_SAMPLES:
        cluster_ids = DBSCAN(
            eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1
        ).fit_predict(vehicle_points)
        for cluster_id in sorted(set(cluster_ids) - {-1}):
            cluster = vehicle_points[cluster_ids == cluster_id]
            if len(cluster) < MIN_CLUSTER_POINTS:
                continue
            low = np.percentile(cluster, 2, axis=0)
            high = np.percentile(cluster, 98, axis=0)
            size = high - low
            if not (
                1.0 <= size[0] <= 18.0
                and 0.8 <= size[1] <= 5.0
                and 0.4 <= size[2] <= 5.0
            ):
                continue
            detections.append({"low": low, "high": high, "points": int(len(cluster))})

    detections.sort(key=lambda detection: detection["points"], reverse=True)
    return detections, threshold, len(all_points), len(sampled), int(len(vehicle_points))


def main():
    for path in (MODEL_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        records = manifest.get("frames", [])
    else:
        records = manifest

    # Use annotated frames with supported classes and existing data files.
    eligible = []
    for record in records:
        annotations = [
            annotation for annotation in record.get("annotations", [])
            if annotation.get("class") in SUPPORTED_CLASSES
        ]
        lidar_path = Path(record.get("lidar_path", ""))
        calibration_path = Path(record.get("calibration_path", ""))
        if annotations and lidar_path.is_file() and calibration_path.is_file():
            eligible.append((record, annotations))

    eligible.sort(key=lambda pair: record_frame_id(pair[0]))
    if MAX_FRAMES is not None:
        eligible = eligible[:MAX_FRAMES]
    if not eligible:
        raise RuntimeError(
            "No eligible annotated frames found. Check manifest paths and the "
            "'class' annotation key in outputs/kitti_vehicle_manifest.json."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if checkpoint.get("feature_count") != 4:
        raise ValueError("Checkpoint does not specify four input features.")
    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold_used = (
        float(checkpoint.get("recommended_threshold", 0.50))
        if THRESHOLD_OVERRIDE is None else float(THRESHOLD_OVERRIDE)
    )
    print("=" * 72)
    print("PROTOTYPE 3D DETECTION EVALUATION — FULL ANNOTATED DATASET")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Eligible annotated frames: {len(eligible):,}")
    print(f"Point threshold: {threshold_used:.2f}")
    print(f"Frames limited by MAX_FRAMES: {MAX_FRAMES}")
    print("Metric: axis-aligned 3D IoU in LiDAR coordinates (diagnostic approximation)")
    print("Processing frames; progress is printed every 25 frames.")

    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for threshold in IOU_THRESHOLDS
    }
    per_frame = []
    skipped = []
    start = time.time()

    for index, (record, annotations) in enumerate(eligible, 1):
        frame_id = record_frame_id(record)
        try:
            lidar_to_camera = read_calibration(record["calibration_path"])
            camera_to_lidar = np.linalg.inv(lidar_to_camera)
            ground_truth = [
                ground_truth_bounds_lidar(annotation, camera_to_lidar)
                for annotation in annotations
            ]
            detections, threshold, original_count, sampled_count, predicted_points = (
                predict_detections(record, checkpoint, model)
            )
            frame_row = {
                "frame_id": frame_id,
                "ground_truth_boxes": len(ground_truth),
                "candidate_boxes": len(detections),
                "original_points": original_count,
                "evaluated_points": sampled_count,
                "predicted_vehicle_points": predicted_points,
                "point_threshold": threshold,
                "results": {},
            }
            for iou_threshold in IOU_THRESHOLDS:
                result = greedy_matches(detections, ground_truth, iou_threshold)
                totals[iou_threshold]["tp"] += result["tp"]
                totals[iou_threshold]["fp"] += result["fp"]
                totals[iou_threshold]["fn"] += result["fn"]
                totals[iou_threshold]["matched_ious"].extend(result["matched_ious"])
                frame_row["results"][str(iou_threshold)] = {
                    key: result[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                }
            per_frame.append(frame_row)
        except Exception as exc:
            skipped.append({"frame_id": frame_id, "error": str(exc)})

        if index % 25 == 0 or index == len(eligible):
            elapsed = time.time() - start
            print(
                f"Progress: {index:,}/{len(eligible):,} frames checked | "
                f"successful: {len(per_frame):,} | skipped: {len(skipped):,} | "
                f"elapsed: {elapsed / 60:.1f} min"
            )

    summary = {}
    for iou_threshold, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        matched_ious = values["matched_ious"]
        summary[str(iou_threshold)] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "full_3d_detection_evaluation.json"
    report = {
        "note": (
            "Prototype metric: axis-aligned 3D IoU in LiDAR coordinates; "
            "not official KITTI oriented 3D detection evaluation."
        ),
        "model_path": str(MODEL_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "device": str(DEVICE),
        "frames_eligible": len(eligible),
        "frames_evaluated": len(per_frame),
        "frames_skipped": len(skipped),
        "point_threshold": threshold_used,
        "iou_summary": summary,
        "per_frame": per_frame,
        "skipped_frames": skipped,
        "settings": {
            "max_frames": MAX_FRAMES,
            "max_points_per_frame": MAX_POINTS,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "min_cluster_points": MIN_CLUSTER_POINTS,
            "supported_classes": sorted(SUPPORTED_CLASSES),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FULL-DATASET SUMMARY")
    print("=" * 72)
    print(f"Frames evaluated: {len(per_frame):,}/{len(eligible):,}")
    print(f"Frames skipped: {len(skipped):,}")
    print(f"Ground-truth boxes: {sum(row['ground_truth_boxes'] for row in per_frame):,}")
    print(f"Candidate boxes: {sum(row['candidate_boxes'] for row in per_frame):,}")
    for iou_threshold, result in summary.items():
        print(f"\nAxis-aligned 3D IoU threshold: {float(iou_threshold):.2f}")
        print(f"  TP={result['tp']:,} | FP={result['fp']:,} | FN={result['fn']:,}")
        print(
            f"  Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | F1={result['f1']:.4f}"
        )
        print(f"  Mean matched IoU={result['mean_matched_iou']:.4f}")
    print(f"\nReport saved to: {report_path.resolve()}")
    print("Reminder: this is a prototype axis-aligned metric, not official KITTI evaluation.")


if __name__ == "__main__":
    main()
