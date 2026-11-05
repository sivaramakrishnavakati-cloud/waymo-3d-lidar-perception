from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle

MODEL_PATH = Path("outputs/lidar_point_classifier_v2.pth")
MANIFEST_PATH = Path("outputs/kitti_vehicle_manifest.json")
OUTPUT_DIR = Path("outputs")
FRAME_ID = "000006"
MAX_POINTS = 100_000
# Start conservatively; adjust after inspecting the resulting clusters.
THRESHOLD_OVERRIDE = None  # e.g. 0.80; None uses the checkpoint's recommendation
DBSCAN_EPS = 1.5           # meters
DBSCAN_MIN_SAMPLES = 8
MIN_CLUSTER_POINTS = 20

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


def main():
    for path in (MODEL_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = next((r for r in manifest
                   if r.get("annotations")
                   and (str(r.get("frame_id", Path(r["lidar_path"]).stem)).zfill(6) == FRAME_ID
                        or Path(r["lidar_path"]).stem == FRAME_ID)
                   and Path(r["lidar_path"]).exists()), None)
    if record is None:
        raise RuntimeError(f"Frame {FRAME_ID} with annotations was not found in {MANIFEST_PATH}.")

    raw = np.fromfile(record["lidar_path"], dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f"Invalid LiDAR file: {record['lidar_path']}")
    all_points = raw.reshape(-1, 4)

    # Match the training/prediction pipeline's input scaling and sampling.
    rng = np.random.default_rng(42)
    if len(all_points) > MAX_POINTS:
        indices = rng.choice(len(all_points), MAX_POINTS, replace=False)
        points = all_points[indices].copy()
    else:
        points = all_points.copy()
    xyz_meters = points[:, :3].astype(np.float64)
    features_np = points.copy()
    features_np[:, :3] /= 50.0

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if checkpoint.get("feature_count") != 4:
        raise ValueError("Checkpoint does not specify four input features.")
    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(features_np).float().to(DEVICE))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    threshold = (float(checkpoint.get("recommended_threshold", 0.50))
                 if THRESHOLD_OVERRIDE is None else THRESHOLD_OVERRIDE)
    predicted = probabilities >= threshold
    vehicle_points = xyz_meters[predicted]
    print("=" * 58)
    print("CLUSTER-BASED LiDAR VEHICLE DETECTION PROTOTYPE")
    print("=" * 58)
    print(f"Frame: {FRAME_ID} | Device: {DEVICE}")
    print(f"Points in original frame: {len(all_points):,}")
    print(f"Points evaluated: {len(points):,}")
    print(f"Threshold: {threshold:.2f} | Predicted vehicle points: {len(vehicle_points):,}")

    clusters = []
    if len(vehicle_points) >= DBSCAN_MIN_SAMPLES:
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit_predict(vehicle_points)
        for cluster_id in sorted(set(labels) - {-1}):
            cluster = vehicle_points[labels == cluster_id]
            if len(cluster) < MIN_CLUSTER_POINTS:
                continue
            low = np.percentile(cluster, 2, axis=0)
            high = np.percentile(cluster, 98, axis=0)
            size = high - low  # x, y, z extents in meters
            # Basic plausibility filter for passenger cars/trucks; not a trained 3D box detector.
            if not (1.0 <= size[0] <= 18.0 and 0.8 <= size[1] <= 5.0 and 0.4 <= size[2] <= 5.0):
                continue
            clusters.append((cluster, low, high, size))

    clusters.sort(key=lambda item: len(item[0]), reverse=True)
    print(f"Candidate clusters retained: {len(clusters)}")
    for i, (_, low, high, size) in enumerate(clusters, 1):
        print(f"Detection {i}: points={len(clusters[i-1][0])}, "
              f"center=({(low[0]+high[0])/2:.1f}, {(low[1]+high[1])/2:.1f}, {(low[2]+high[2])/2:.1f}) m, "
              f"size=(Lx={size[0]:.1f}, Wy={size[1]:.1f}, Hz={size[2]:.1f}) m")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="black")
    ax.set_facecolor("black")
    background = ~predicted
    ax.scatter(xyz_meters[background, 0], xyz_meters[background, 1], s=0.25,
               c="#00ff35", linewidths=0, label="Predicted background")
    ax.scatter(vehicle_points[:, 0], vehicle_points[:, 1], s=1.0,
               c="#ffb000", linewidths=0, label="Predicted vehicle points")
    for i, (cluster, low, high, size) in enumerate(clusters, 1):
        rect = Rectangle((low[0], low[1]), size[0], size[1], fill=False,
                         edgecolor="#00bfff", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(low[0], high[1], str(i), color="white", fontsize=8)
    ax.set_xlabel("LiDAR X (meters)", color="white")
    ax.set_ylabel("LiDAR Y (meters)", color="white")
    ax.set_title(f"Cluster-Based LiDAR Vehicle Candidates — Frame {FRAME_ID}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(facecolor="black", edgecolor="#666666")
    for item in legend.get_texts():
        item.set_color("white")
    fig.tight_layout()
    output_path = OUTPUT_DIR / f"lidar_vehicle_clusters_{FRAME_ID}.png"
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    print(f"Visualization saved to: {output_path.resolve()}")
    print("Note: these are axis-aligned cluster proposals, not verified 3D vehicle detections.")
    plt.show()


if __name__ == "__main__":
    main()
