
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
OUTPUT_DIR = Path("outputs/bev_detector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_FRAMES = 500
NUM_POINTS = 16384
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-3

# KITTI LiDAR coordinates: X forward, Y left, Z up.
X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -40.0, 40.0
GRID_SIZE = 0.5

GRID_H = int((X_MAX - X_MIN) / GRID_SIZE)
GRID_W = int((Y_MAX - Y_MIN) / GRID_SIZE)

CLASS_NAMES = {"Car": 0, "Van": 1, "Truck": 2}
NUM_CLASSES = len(CLASS_NAMES)

# Output channels:
# 0: objectness
# 1-3: class logits
# 4-6: fractional cell X, fractional cell Y, center Z
# 7-9: log(length), log(width), log(height)
# 10-11: sin(yaw), cos(yaw)
OUTPUT_CHANNELS = 12

# Model output is downsampled by 4 (two 2x pooling layers).
# Create targets directly on that grid so object labels are not lost.
OUTPUT_SCALE = 4
OUTPUT_GRID_H = GRID_H // OUTPUT_SCALE
OUTPUT_GRID_W = GRID_W // OUTPUT_SCALE
OUTPUT_CELL_SIZE = GRID_SIZE * OUTPUT_SCALE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MANIFEST AND CALIBRATION
# ============================================================

def load_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, raw = line.split(":", 1)

            try:
                values[key.strip()] = np.array(
                    [float(v) for v in raw.split()],
                    dtype=np.float64
                )
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing required calibration values: {path}")

    if values["Tr_velo_to_cam"].size != 12:
        raise ValueError(f"Invalid Tr_velo_to_cam in {path}")

    if values["R0_rect"].size != 9:
        raise ValueError(f"Invalid R0_rect in {path}")

    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)

    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)

    # LiDAR -> rectified camera.
    return rect @ velo_to_cam


def get_path(record, key, fallback):
    value = record.get(key)
    if value:
        return Path(value)

    return fallback


# ============================================================
# KITTI BOX CONVERSION
# ============================================================

def camera_box_to_lidar(annotation, camera_to_lidar):
    """
    Convert one KITTI camera-coordinate box to LiDAR coordinates.

    KITTI label dimensions are H, W, L.
    KITTI location is the bottom center in camera coordinates.
    Camera axes: X right, Y down, Z forward.
    LiDAR axes: X forward, Y left, Z up.
    """

    dims = annotation.get(
        "dimensions_hwl",
        annotation.get("dimensions")
    )

    location = annotation.get(
        "location_camera",
        annotation.get("location")
    )

    yaw_camera = annotation.get(
        "rotation_y",
        annotation.get("yaw")
    )

    if dims is None or location is None or yaw_camera is None:
        return None

    height, width, length = map(float, dims)
    x, y, z = map(float, location)

    if min(height, width, length) <= 0:
        return None

    center_camera = np.array(
        [x, y, z, 1.0], dtype=np.float64
    )

    center_lidar = (camera_to_lidar @ center_camera)[:3]

    # Build the camera-coordinate box's forward direction,
    # then transform that direction into LiDAR coordinates.
    forward_camera = np.array(
        [math.cos(yaw_camera), 0.0, -math.sin(yaw_camera)],
        dtype=np.float64
    )

    rotation_camera_to_lidar = camera_to_lidar[:3, :3]
    forward_lidar = rotation_camera_to_lidar @ forward_camera

    # KITTI yaw is defined in camera coordinates. Convert it
    # to a LiDAR-frame yaw using the transformed heading.
    yaw_lidar = math.atan2(forward_lidar[1], forward_lidar[0])

    # Camera location is at the bottom center. Convert to
    # the geometric center of the box in LiDAR coordinates.
    center_lidar[2] += height / 2.0

    return {
        "class_id": int(annotation["class_id"]),
        "center": center_lidar.astype(np.float32),
        "length": length,
        "width": width,
        "height": height,
        "yaw": float(yaw_lidar),
    }


# ============================================================
# DATASET
# ============================================================

class KITTIBeVDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    @staticmethod
    def make_bev(points):
        bev = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)

        valid = (
            (points[:, 0] >= X_MIN)
            & (points[:, 0] < X_MAX)
            & (points[:, 1] >= Y_MIN)
            & (points[:, 1] < Y_MAX)
            & np.isfinite(points[:, :3]).all(axis=1)
        )

        points = points[valid]

        if len(points):
            ix = ((points[:, 0] - X_MIN) / GRID_SIZE).astype(np.int32)
            iy = ((points[:, 1] - Y_MIN) / GRID_SIZE).astype(np.int32)

            np.maximum.at(
                bev[0],
                (ix, iy),
                np.clip((points[:, 2] + 3.0) / 6.0, 0.0, 1.0)
            )

            np.add.at(bev[1], (ix, iy), 1.0)
            bev[1] = np.log1p(bev[1]) / np.log(65.0)

        return bev

    def __getitem__(self, index):
        record = self.records[index]

        lidar_path = Path(record["lidar_path"])
        calib_path = Path(record["calibration_path"])

        if not lidar_path.is_file():
            raise FileNotFoundError(lidar_path)

        if not calib_path.is_file():
            raise FileNotFoundError(calib_path)

        raw = np.fromfile(lidar_path, dtype=np.float32)

        if raw.size == 0 or raw.size % 4 != 0:
            raise ValueError(f"Invalid LiDAR data: {lidar_path}")

        points = raw.reshape(-1, 4)
        bev = self.make_bev(points)

        lidar_to_camera = read_calibration(calib_path)
        camera_to_lidar = np.linalg.inv(lidar_to_camera)

        target = np.zeros(
            (OUTPUT_CHANNELS, OUTPUT_GRID_H, OUTPUT_GRID_W),
            dtype=np.float32
        )

        annotations = record.get("annotations", [])
        boxes = []

        for annotation in annotations:
            cls = annotation.get("class", annotation.get("type", ""))

            if cls not in CLASS_NAMES:
                continue

            normalized_annotation = dict(annotation)
            normalized_annotation["class_id"] = CLASS_NAMES[cls]

            box = camera_box_to_lidar(
                normalized_annotation,
                camera_to_lidar
            )

            if box is None:
                continue

            cx, cy, cz = box["center"]

            if not (
                X_MIN <= cx < X_MAX
                and Y_MIN <= cy < Y_MAX
            ):
                continue

            ix = int((cx - X_MIN) / OUTPUT_CELL_SIZE)
            iy = int((cy - Y_MIN) / OUTPUT_CELL_SIZE)

            # A single cell can encode only one box in this
            # baseline. Keep the first box assigned to a cell.
            if target[0, ix, iy] > 0:
                continue

            target[0, ix, iy] = 1.0
            target[1 + box["class_id"], ix, iy] = 1.0

            target[4, ix, iy] = (cx - X_MIN) / OUTPUT_CELL_SIZE - ix
            target[5, ix, iy] = (cy - Y_MIN) / OUTPUT_CELL_SIZE - iy
            target[6, ix, iy] = cz

            target[7, ix, iy] = math.log(box["length"])
            target[8, ix, iy] = math.log(box["width"])
            target[9, ix, iy] = math.log(box["height"])

            target[10, ix, iy] = math.sin(box["yaw"])
            target[11, ix, iy] = math.cos(box["yaw"])

            boxes.append(box)

        return {
            "bev": torch.from_numpy(bev),
            "target": torch.from_numpy(target),
            "frame_id": str(record.get("frame_id", index)),
            "box_count": len(boxes),
        }


# ============================================================
# MODEL
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

        self.head = nn.Conv2d(64, OUTPUT_CHANNELS, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# LOSS
# ============================================================


def detection_loss(prediction, target):
    """
    Calculate BEV detection loss.

    prediction: [B, 12, H_out, W_out]
    target:     [B, 12, H_out, W_out]

    Channels:
      0       : objectness
      1:4     : class targets (4 classes)
      4:12    : regression targets (8 values)
    """
    # Targets are already built at output resolution; do not resize them.
    target = target.float()
    if target.shape[-2:] != prediction.shape[-2:]:
        raise ValueError(
            f"Target grid {tuple(target.shape[-2:])} does not match "
            f"prediction grid {tuple(prediction.shape[-2:])}."
        )

    # Objectness
    objectness_logits = prediction[:, 0]
    objectness_target = target[:, 0].clamp(0.0, 1.0)

    positives = objectness_target.sum().clamp(min=1.0)
    negatives = (1.0 - objectness_target).sum().clamp(min=1.0)
    positive_weight = (negatives / positives).clamp(max=50.0)

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness_logits,
        objectness_target,
        pos_weight=positive_weight,
    )

    # Classification
    positive_mask = objectness_target > 0.5

    if positive_mask.any():
        class_logits = prediction[:, 1:4].permute(0, 2, 3, 1)
        class_targets = target[:, 1:4].argmax(dim=1).long()

        class_loss = F.cross_entropy(
            class_logits[positive_mask],
            class_targets[positive_mask],
        )

        # Regression
        regression_prediction = (
            prediction[:, 4:12].permute(0, 2, 3, 1)
        )
        regression_target = (
            target[:, 4:12].permute(0, 2, 3, 1)
        )

        regression_loss = F.smooth_l1_loss(
            regression_prediction[positive_mask],
            regression_target[positive_mask],
        )
    else:
        class_loss = prediction.sum() * 0.0
        regression_loss = prediction.sum() * 0.0

    total_loss = (
        objectness_loss
        + class_loss
        + 2.0 * regression_loss
    )

    return total_loss, {
        "objectness": objectness_loss.detach(),
        "classification": class_loss.detach(),
        "regression": regression_loss.detach(),
    }


# ============================================================
# TRAINING
# ============================================================

def main():
    print("=" * 65)
    print("BEV VEHICLE DETECTOR — TRAINING")
    print("=" * 65)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    records = load_manifest(MANIFEST_PATH)

    # Filter frames with the expected manifest fields.
    valid_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        if not record.get("lidar_path") or not record.get("calibration_path"):
            continue

        if not record.get("annotations"):
            continue

        valid_records.append(record)

    random.shuffle(valid_records)
    valid_records = valid_records[:MAX_FRAMES]

    if len(valid_records) < 10:
        raise RuntimeError(
            f"Only {len(valid_records)} usable records found. "
            "At least 10 are required for this experiment."
        )

    # Split by frame before training.
    split_index = max(1, int(0.8 * len(valid_records)))
    train_records = valid_records[:split_index]
    val_records = valid_records[split_index:]

    train_dataset = KITTIBeVDataset(train_records)
    val_dataset = KITTIBeVDataset(val_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = BEVDetector().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    print(f"Training frames: {len(train_dataset)}")
    print(f"Validation frames: {len(val_dataset)}")
    print(f"BEV grid: {GRID_H} x {GRID_W}")
    print(f"Output channels: {OUTPUT_CHANNELS}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify one real batch before training.
    first_batch = next(iter(train_loader))

    print("Input batch:", tuple(first_batch["bev"].shape))
    print("Target batch:", tuple(first_batch["target"].shape))
    print("Annotated boxes in first batch:", first_batch["box_count"])
    
    
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        
        model.train()
        train_total = 0.0

        for batch in train_loader:
            bev = batch["bev"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(bev)
            loss, components = detection_loss(prediction, target)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_total += loss.item()

        train_average = train_total / max(1, len(train_loader))

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                bev = batch["bev"].to(DEVICE)
                target = batch["target"].to(DEVICE)

                prediction = model(bev)
                loss, _ = detection_loss(prediction, target)
                val_total += loss.item()

        val_average = val_total / max(1, len(val_loader))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_average:.4f} | "
            f"validation loss={val_average:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_average,
            "validation_loss": val_average,
            "grid_config": {
                "x_min": X_MIN,
                "x_max": X_MAX,
                "y_min": Y_MIN,
                "y_max": Y_MAX,
                "grid_size": GRID_SIZE,
            },
            "class_names": CLASS_NAMES,
        }

        # Always save the latest checkpoint.
        latest_path = OUTPUT_DIR / "bev_detector_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save the best checkpoint when validation loss improves.
        if val_average < best_val_loss:
            best_val_loss = val_average

            best_path = OUTPUT_DIR / "bev_detector_best.pt"
            torch.save(checkpoint, best_path)

            print(f"New best model saved! Validation loss: {best_val_loss:.4f}")


    print("\nTraining run completed.")
    print("Checkpoint:", OUTPUT_DIR / "bev_detector_latest.pt")
    print(
        "Reminder: falling loss is not sufficient to establish "
        "good 3D detection. We must evaluate decoded boxes using "
        "3D IoU and precision/recall next."
    )


if __name__ == "__main__":
    main()

