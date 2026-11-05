from pathlib import Path
import json
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
OUTPUT_PATH = Path("outputs/lidar_point_classifier_v2.pth")
NUM_POINTS = 4096
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_lidar(path):
    points = np.fromfile(path, dtype=np.float32)
    if points.size == 0 or points.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR file: {path}")
    return points.reshape(-1, 4)


class LidarVehicleDataset(Dataset):
    """Point-level labels: 0=background, 1=inside a KITTI vehicle box."""

    def __init__(self, manifest_path):
        with manifest_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        self.records = [r for r in records if r.get("annotations")]
        if not self.records:
            raise RuntimeError("No annotated frames found in the manifest.")
        print(f"Annotated frames loaded: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def read_calibration(path):
        values = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
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
            raise ValueError(f"Missing calibration values in {path}")

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
        rectification = np.eye(4, dtype=np.float64)
        rectification[:3, :3] = values["R0_rect"].reshape(3, 3)
        return rectification @ transform

    @staticmethod
    def points_inside_box(points_camera, annotation):
        height, width, length = annotation["dimensions_hwl"]
        x, y, z = annotation["location_camera"]
        yaw = annotation["rotation_y"]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        center = np.array([x, y, z], dtype=np.float64)
        local = (rotation.T @ (points_camera - center).T).T

        return (
            (np.abs(local[:, 0]) <= length / 2)
            & (local[:, 1] <= 0.1)
            & (local[:, 1] >= -height - 0.1)
            & (np.abs(local[:, 2]) <= width / 2)
        )

    def __getitem__(self, index):
        record = self.records[index]
        points = load_lidar(record["lidar_path"])
        lidar_xyz = points[:, :3].astype(np.float64)

        lidar_to_camera = self.read_calibration(record["calibration_path"])
        homogeneous = np.column_stack([lidar_xyz, np.ones(len(lidar_xyz))])
        points_camera = (lidar_to_camera @ homogeneous.T).T[:, :3]

        targets = np.zeros(len(points), dtype=np.float32)
        for annotation in record["annotations"]:
            targets[self.points_inside_box(points_camera, annotation)] = 1.0

        # Random point sampling limits memory and exposes different points over epochs.
        if len(points) >= NUM_POINTS:
            indices = np.random.choice(len(points), NUM_POINTS, replace=False)
        else:
            indices = np.random.choice(len(points), NUM_POINTS, replace=True)

        sampled = points[indices].copy()
        sampled_targets = targets[indices]

        # Keep the same 4 input features and coordinate scaling as the existing predictor.
        sampled[:, :3] /= 50.0
        features = torch.from_numpy(sampled).float()
        labels = torch.from_numpy(sampled_targets).float()
        return features, labels


class PointClassifier(nn.Module):
    # Keep this architecture compatible with the existing 4-feature predictor.
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, points):
        return self.network(points).squeeze(-1)


def metrics_at_threshold(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    actual = labels >= 0.5
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold), "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. "
            "Run prepare_kitti_dataset.py first."
        )

    print("=" * 64)
    print("LIDAR VEHICLE POINT CLASSIFIER — IMPROVED TRAINING")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = LidarVehicleDataset(MANIFEST_PATH)
    if len(dataset) < 2:
        raise RuntimeError("At least two annotated frames are required.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = PointClassifier().to(DEVICE)

    # Estimate class balance from a deterministic pass over training samples.
    # Clamp the weight to avoid excessively large gradients on sparse positives.
    positive = 0
    total = 0
    for _, labels in train_loader:
        positive += int(labels.sum().item())
        total += labels.numel()
    negative = total - positive
    pos_weight_value = float(np.clip(negative / max(positive, 1), 1.0, 10.0))
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    print(f"Training frames: {train_size} | Validation frames: {val_size}")
    print(f"Positive class weight (clamped): {pos_weight_value:.3f}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels_device = labels.to(DEVICE)
                logits = model(features)
                val_loss_sum += criterion(logits, labels_device).item()
                all_probabilities.append(torch.sigmoid(logits).cpu().numpy().ravel())
                all_labels.append(labels.numpy().ravel())

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        probabilities = np.concatenate(all_probabilities)
        labels_np = np.concatenate(all_labels)

        # Report both conventional 0.50 metrics and the best F1 threshold
        # on this validation split. Threshold is diagnostic, not a guarantee.
        baseline = metrics_at_threshold(probabilities, labels_np, 0.50)
        threshold_results = [
            metrics_at_threshold(probabilities, labels_np, t)
            for t in np.arange(0.10, 0.951, 0.05)
        ]
        best_f1 = max(threshold_results, key=lambda m: m["f1"])
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"F1@0.50={baseline['f1']:.4f} | "
            f"P@0.50={baseline['precision']:.4f} | "
            f"R@0.50={baseline['recall']:.4f} | "
            f"best_val_F1={best_f1['f1']:.4f}@{best_f1['threshold']:.2f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping: validation loss did not improve for {PATIENCE} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    # Recompute final validation metrics from the best-loss checkpoint.
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            logits = model(features.to(DEVICE))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy().ravel())
            all_labels.append(labels.numpy().ravel())

    probabilities = np.concatenate(all_probabilities)
    labels_np = np.concatenate(all_labels)
    threshold_results = [
        metrics_at_threshold(probabilities, labels_np, t)
        for t in np.arange(0.10, 0.951, 0.05)
    ]
    best_f1 = max(threshold_results, key=lambda m: m["f1"])
    at_050 = metrics_at_threshold(probabilities, labels_np, 0.50)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_points": NUM_POINTS,
        "feature_count": 4,
        "classes": ["background", "vehicle"],
        "recommended_threshold": best_f1["threshold"],
        "validation_metrics_at_050": {
            k: at_050[k] for k in ("accuracy", "precision", "recall", "f1")
        },
        "validation_best_f1_metrics": best_f1,
        "training_config": {
            "epochs_requested": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_class_weight": pos_weight_value,
            "seed": SEED
        }
    }, OUTPUT_PATH)

    print("\n--- Final validation metrics (best validation-loss checkpoint) ---")
    print(
        f"At threshold 0.50: accuracy={at_050['accuracy']:.4f}, "
        f"precision={at_050['precision']:.4f}, recall={at_050['recall']:.4f}, "
        f"F1={at_050['f1']:.4f}"
    )
    print(
        f"Best tested F1 threshold={best_f1['threshold']:.2f}: "
        f"precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}, "
        f"F1={best_f1['f1']:.4f}"
    )
    print(f"New model saved to: {OUTPUT_PATH.resolve()}")
    print("Existing outputs/lidar_point_classifier.pth was not overwritten.")
    print("Training complete.")


if __name__ == "__main__":
    main()
