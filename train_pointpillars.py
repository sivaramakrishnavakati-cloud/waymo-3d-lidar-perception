
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "kitti_vehicle_manifest.json"

NUM_POINTS = 16384
BATCH_SIZE = 2

CLASS_NAMES = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
}


# --------------------------------------------------
# KITTI Dataset
# --------------------------------------------------

class KITTIVehicleDataset(Dataset):
    def __init__(self, manifest_path, num_points=NUM_POINTS):
        self.manifest_path = Path(manifest_path)
        self.num_points = num_points

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}\n"
                "Run prepare_kitti_dataset.py first."
            )

        with self.manifest_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        # Keep records with at least one supported vehicle annotation.
        self.records = [
            record
            for record in records
            if record.get("annotations")
        ]

        if not self.records:
            raise RuntimeError(
                "No annotated vehicle frames were found in the manifest."
            )

        print(f"Manifest: {self.manifest_path}")
        print(f"Frames with vehicle annotations: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def load_lidar(path):
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"LiDAR file not found: {path}")

        raw = np.fromfile(path, dtype=np.float32)

        if raw.size == 0 or raw.size % 4 != 0:
            raise ValueError(f"Invalid KITTI LiDAR file: {path}")

        # KITTI point format: x, y, z, reflectance.
        return raw.reshape(-1, 4)

    @staticmethod
    def load_calibration(path):
        values = {}

        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)

                try:
                    values[key.strip()] = np.asarray(
                        [float(item) for item in value.split()],
                        dtype=np.float64,
                    )
                except ValueError:
                    continue

        if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
            raise ValueError(f"Missing calibration matrices: {path}")

        if values["Tr_velo_to_cam"].size != 12:
            raise ValueError(f"Invalid Tr_velo_to_cam matrix: {path}")

        if values["R0_rect"].size != 9:
            raise ValueError(f"Invalid R0_rect matrix: {path}")

        velo_to_camera = np.eye(4, dtype=np.float64)
        velo_to_camera[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)

        rectification = np.eye(4, dtype=np.float64)
        rectification[:3, :3] = values["R0_rect"].reshape(3, 3)

        return rectification @ velo_to_camera

    @staticmethod
    def points_inside_box(points_camera, annotation):
        height, width, length = annotation["dimensions_hwl"]
        x, y, z = annotation["location_camera"]
        yaw = annotation["rotation_y"]

        center = np.asarray([x, y, z], dtype=np.float64)

        c = np.cos(yaw)
        s = np.sin(yaw)

        rotation = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])

        local = (rotation.T @ (points_camera - center).T).T

        return (
            (np.abs(local[:, 0]) <= length / 2)
            & (local[:, 1] <= 0.1)
            & (local[:, 1] >= -height - 0.1)
            & (np.abs(local[:, 2]) <= width / 2)
        )

    def __getitem__(self, index):
        record = self.records[index]

        points = self.load_lidar(record["lidar_path"])
        lidar_xyz = points[:, :3].astype(np.float64)

        # Convert LiDAR points to rectified camera coordinates.
        lidar_to_camera = self.load_calibration(
            record["calibration_path"]
        )

        homogeneous = np.column_stack([
            lidar_xyz,
            np.ones(len(lidar_xyz), dtype=np.float64),
        ])

        points_camera = (
            lidar_to_camera @ homogeneous.T
        ).T[:, :3]

        # Diagnostic point-level targets.
        # These are NOT the final 3D detection targets.
        point_targets = np.zeros(len(points), dtype=np.float32)

        for annotation in record["annotations"]:
            inside = self.points_inside_box(
                points_camera,
                annotation,
            )
            point_targets[inside] = 1.0

        # Sample a fixed number of points for the next stage.
        if len(points) >= self.num_points:
            indices = np.random.choice(
                len(points),
                self.num_points,
                replace=False,
            )
        else:
            indices = np.random.choice(
                len(points),
                self.num_points,
                replace=True,
            )

        sampled_points = points[indices].copy()
        sampled_targets = point_targets[indices].copy()

        return {
            "points": torch.from_numpy(sampled_points),
            "point_targets": torch.from_numpy(sampled_targets),
            "frame_id": record["frame_id"],
            "annotations": record["annotations"],
        }


# --------------------------------------------------
# Dataset verification
# --------------------------------------------------

def main():
    dataset = KITTIVehicleDataset(MANIFEST_PATH)

    # Test one sample before creating a training loop.
    sample = dataset[0]

    print("\nDataset sample verification")
    print("-" * 40)
    print("Frame ID:", sample["frame_id"])
    print("Point tensor shape:", tuple(sample["points"].shape))
    print("Point target shape:", tuple(sample["point_targets"].shape))
    print("Annotations:", len(sample["annotations"]))
    print("Positive sampled points:",
          int(sample["point_targets"].sum().item()))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda samples: {
            "points": torch.stack([s["points"] for s in samples]),
            "point_targets": torch.stack(
                [s["point_targets"] for s in samples]
            ),
            "frame_id": [s["frame_id"] for s in samples],
            "annotations": [s["annotations"] for s in samples],
        },
    )

    batch = next(iter(loader))

    print("\nDataLoader verification")
    print("-" * 40)
    print("Batch point shape:", tuple(batch["points"].shape))
    print("Batch target shape:", tuple(batch["point_targets"].shape))
    print("\nDataset loading successful.")


if __name__ == "__main__":
    main()
