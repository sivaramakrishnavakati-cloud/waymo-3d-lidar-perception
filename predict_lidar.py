
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = Path("outputs/lidar_point_classifier.pth")
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_POINTS = 100_000


# --------------------------------------------------
# Model architecture — must match training
# --------------------------------------------------
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


# --------------------------------------------------
# Main prediction
# --------------------------------------------------
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = __import__("json").loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    records = [
        record for record in manifest
        if record.get("annotations")
        and Path(record["lidar_path"]).exists()
    ]

    if not records:
        raise RuntimeError(
            "No annotated LiDAR frames found. Check the manifest paths."
        )

    # Use the first annotated frame for the initial test.
    record = records[0]
    lidar_path = Path(record["lidar_path"])

    points = np.fromfile(lidar_path, dtype=np.float32)

    if points.size == 0 or points.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR file: {lidar_path}")

    points = points.reshape(-1, 4)
    original_points = points[:, :3].copy()

    # The training script normalizes XYZ by dividing by 50.
    points[:, :3] /= 50.0

    # Reproduce the training script's four input features.
    if len(points) >= MAX_POINTS:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(points), MAX_POINTS, replace=False
        )
        points = points[indices]
        original_points = original_points[indices]

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )

    if checkpoint.get("feature_count") != 4:
        raise ValueError("The saved model does not specify four input features.")

    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = torch.from_numpy(points).float().to(DEVICE)

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    THRESHOLD = 0.90
    predictions = probabilities >= THRESHOLD

    print("=" * 55)
    print("LIDAR VEHICLE POINT PREDICTIONS")
    print("=" * 55)
    print(f"Device: {DEVICE}")
    print(f"Frame: {record.get('frame_id', lidar_path.stem)}")
    print(f"Points tested: {len(points):,}")
    print(f"Predicted vehicle points: {predictions.sum():,}")
    print(f"Predicted background points: {(~predictions).sum():,}")
    print(f"Mean vehicle probability: {probabilities.mean():.4f}")

    # Plot predicted points in a bird's-eye view.
    plt.figure(figsize=(12, 9))

    background = ~predictions
    plt.scatter(
        original_points[background, 0],
        original_points[background, 1],
        s=0.3,
        label="Predicted background",
    )

    plt.scatter(
        original_points[predictions, 0],
        original_points[predictions, 1],
        s=1.0,
        label="Predicted vehicle",
    )

    plt.xlabel("LiDAR X (meters)")
    plt.ylabel("LiDAR Y (meters)")
    plt.title("LiDAR Point Classification — Bird's-Eye View")
    plt.axis("equal")
    plt.legend(markerscale=5)
    plt.tight_layout()

    output_path = Path("outputs/lidar_predictions.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Visualization saved to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
