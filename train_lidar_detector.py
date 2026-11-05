from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
NUM_POINTS = 4096
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# Load a LiDAR point cloud
# --------------------------------------------------
def load_lidar(path):
    points = np.fromfile(path, dtype=np.float32)

    if points.size == 0 or points.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR file: {path}")

    points = points.reshape(-1, 4)
    return points


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class LidarVehicleDataset(Dataset):
    """
    Point-level baseline:
    0 = background point
    1 = point inside an annotated vehicle box

    Vehicle boxes are converted from camera coordinates
    to LiDAR coordinates using the frame calibration.
    """

    def __init__(self, manifest_path):
        with manifest_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        self.records = [
            record for record in records
            if record["annotations"]
        ]

        if not self.records:
            raise RuntimeError("No annotated frames found.")

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
                        [float(v) for v in raw.split()],
                        dtype=np.float64
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

        # KITTI location is the bottom center of the box.
        center = np.array([x, y, z])
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

        lidar_to_camera = self.read_calibration(
            record["calibration_path"]
        )

        homogeneous = np.column_stack([
            lidar_xyz,
            np.ones(len(lidar_xyz))
        ])

        points_camera = (lidar_to_camera @ homogeneous.T).T[:, :3]

        # Create point-level targets from annotated 3D boxes.
        targets = np.zeros(len(points), dtype=np.float32)

        for annotation in record["annotations"]:
            inside = self.points_inside_box(
                points_camera, annotation
            )
            targets[inside] = 1.0

        # Keep the input size manageable.
        if len(points) >= NUM_POINTS:
            indices = np.random.choice(
                len(points), NUM_POINTS, replace=False
            )
        else:
            indices = np.random.choice(
                len(points), NUM_POINTS, replace=True
            )

        sampled = points[indices].copy()
        sampled_targets = targets[indices]

        # Normalize coordinates and include reflectance.
        sampled[:, :3] /= 50.0

        features = torch.from_numpy(sampled).float()
        labels = torch.from_numpy(sampled_targets).float()

        return features, labels


# --------------------------------------------------
# Point-level neural network
# --------------------------------------------------
class PointClassifier(nn.Module):
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


# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------
def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. "
            "Run prepare_kitti_dataset.py first."
        )

    print("=" * 55)
    print("LIDAR VEHICLE POINT CLASSIFICATION")
    print("=" * 55)
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = LidarVehicleDataset(MANIFEST_PATH)

    if len(dataset) < 2:
        raise RuntimeError("At least two annotated frames are required.")

    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(SEED)

    train_set, val_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )

    model = PointClassifier().to(DEVICE)

    # Point-level labels are imbalanced, so weight positive examples.
    positive = 0
    total = 0

    for _, labels in train_loader:
        positive += int(labels.sum().item())
        total += labels.numel()

    negative = total - positive
    positive_weight = torch.tensor(
        [max(1.0, negative / max(positive, 1))],
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total_points = 0
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(features)
                val_loss += criterion(logits, labels).item()

                predictions = logits >= 0
                correct += (predictions == (labels >= 0.5)).sum().item()
                total_points += labels.numel()

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss / len(train_loader):.4f} | "
            f"Validation loss: {val_loss / len(val_loader):.4f} | "
            f"Point accuracy: {correct / total_points:.4f}"
        )

    output_path = Path("outputs/lidar_point_classifier.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_points": NUM_POINTS,
        "feature_count": 4,
        "classes": ["background", "vehicle"]
    }, output_path)

    print(f"\nModel saved to: {output_path.resolve()}")
    print("Training complete.")


if __name__ == "__main__":
    main()