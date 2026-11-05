from pathlib import Path
from collections import Counter
import json

# Project configuration
DATA_ROOT = Path("data/kitti")
TRAINING_DIR = DATA_ROOT / "training"

VELODYNE_DIR = TRAINING_DIR / "velodyne"
LABEL_DIR = TRAINING_DIR / "label_2"
CALIB_DIR = TRAINING_DIR / "calib"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASSES = {"Car", "Van", "Truck"}


def read_labels(label_path):
    """Read KITTI 3D annotations and retain supported vehicle classes."""
    annotations = []

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()

            if len(fields) < 15:
                continue

            class_name = fields[0]

            if class_name not in VEHICLE_CLASSES:
                continue

            try:
                height, width, length = map(float, fields[8:11])
                x, y, z = map(float, fields[11:14])
                rotation_y = float(fields[14])
            except ValueError:
                continue

            if height <= 0 or width <= 0 or length <= 0:
                continue

            annotations.append({
                "class": class_name,
                "dimensions_hwl": [height, width, length],
                "location_camera": [x, y, z],
                "rotation_y": rotation_y,
            })

    return annotations


def main():
    for directory in (VELODYNE_DIR, LABEL_DIR, CALIB_DIR):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required dataset directory not found: {directory}"
            )

    lidar_files = sorted(VELODYNE_DIR.glob("*.bin"))
    label_files = sorted(LABEL_DIR.glob("*.txt"))
    calibration_files = sorted(CALIB_DIR.glob("*.txt"))

    lidar_ids = {p.stem for p in lidar_files}
    label_ids = {p.stem for p in label_files}
    calibration_ids = {p.stem for p in calibration_files}

    common_ids = sorted(lidar_ids & label_ids & calibration_ids)

    print("=" * 55)
    print("KITTI 3D PERCEPTION — DATASET VALIDATION")
    print("=" * 55)
    print(f"LiDAR frames:       {len(lidar_files)}")
    print(f"Label files:        {len(label_files)}")
    print(f"Calibration files:  {len(calibration_files)}")
    print(f"Matched frames:     {len(common_ids)}")

    if not common_ids:
        raise RuntimeError("No matching LiDAR, label, and calibration frames.")

    class_counts = Counter()
    frames_with_vehicles = 0
    total_annotations = 0
    records = []

    for frame_id in common_ids:
        label_path = LABEL_DIR / f"{frame_id}.txt"
        annotations = read_labels(label_path)

        if annotations:
            frames_with_vehicles += 1

        total_annotations += len(annotations)

        for annotation in annotations:
            class_counts[annotation["class"]] += 1

        records.append({
            "frame_id": frame_id,
            "lidar_path": str(VELODYNE_DIR / f"{frame_id}.bin"),
            "label_path": str(label_path),
            "calibration_path": str(CALIB_DIR / f"{frame_id}.txt"),
            "annotations": annotations,
        })

    manifest_path = OUTPUT_DIR / "kitti_vehicle_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    print(f"\nFrames with supported vehicles: {frames_with_vehicles}")
    print(f"Total vehicle annotations:      {total_annotations}")
    print("\nVehicle annotation counts:")

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    print(f"\nManifest saved to: {manifest_path}")
    print("\nDataset validation completed.")


if __name__ == "__main__":
    main()
