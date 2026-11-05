
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = Path("outputs/lidar_point_classifier_v2.pth")
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_POINTS = 100_000
FRAME_ID = "000011"  # Select the KITTI frame to visualize


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

    # Select the requested frame instead of silently using the first manifest entry.
    record = next(
        (
            item for item in records
            if str(item.get("frame_id", Path(item["lidar_path"]).stem)).zfill(6)
            == FRAME_ID
            or Path(item["lidar_path"]).stem == FRAME_ID
        ),
        None,
    )
    if record is None:
        available = [
            str(item.get("frame_id", Path(item["lidar_path"]).stem))
            for item in records[:20]
        ]
        raise RuntimeError(
            f"Frame {FRAME_ID} was not found among annotated frames in the manifest. "
            f"First available frame IDs: {available}. "
            "Check outputs/kitti_vehicle_manifest.json and its LiDAR paths."
        )
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

    THRESHOLD = float(checkpoint.get("recommended_threshold", 0.80))
    predictions = probabilities >= THRESHOLD

    print(f"Prediction threshold: {THRESHOLD:.2f}")

    print("=" * 55)
    print("LIDAR VEHICLE POINT PREDICTIONS")
    print("=" * 55)
    print(f"Device: {DEVICE}")
    print(f"Frame: {record.get('frame_id', lidar_path.stem)}")
    print(f"Points tested: {len(points):,}")
    print(f"Predicted vehicle points: {predictions.sum():,}")
    print(f"Predicted background points: {(~predictions).sum():,}")
    print(f"Mean vehicle probability: {probabilities.mean():.4f}")

    # Bird's-eye visualization: black background, green LiDAR,
    # orange/yellow points classified as vehicle.
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="black")
    ax.set_facecolor("black")

    background = ~predictions
    ax.scatter(
        original_points[background, 0],
        original_points[background, 1],
        s=0.35,
        c="#00ff35",
        linewidths=0,
        label="Predicted background / LiDAR",
    )

    ax.scatter(
        original_points[predictions, 0],
        original_points[predictions, 1],
        s=1.2,
        c="#ffb000",
        linewidths=0,
        label="Predicted vehicle points",
    )

    ax.set_xlabel("LiDAR X (meters)", color="white")
    ax.set_ylabel("LiDAR Y (meters)", color="white")
    ax.set_title(f"LiDAR Point Classification — Frame {FRAME_ID}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(markerscale=5, facecolor="black", edgecolor="#666666")
    for text_item in legend.get_texts():
        text_item.set_color("white")
    fig.tight_layout()

    output_path = Path(f"outputs/lidar_predictions_{FRAME_ID}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Visualization saved to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
