import json
from collections import Counter
from pathlib import Path

manifest_path = Path("outputs/kitti_vehicle_manifest.json")

if not manifest_path.exists():
    raise FileNotFoundError(
        f"Manifest not found: {manifest_path}. "
        "Run prepare_kitti_dataset.py first."
    )

with manifest_path.open("r", encoding="utf-8") as file:
    records = json.load(file)

class_counts = Counter()
objects_per_frame = Counter()
depths = []

for record in records:
    annotations = record.get("annotations", [])
    objects_per_frame[len(annotations)] += 1

    for annotation in annotations:
        class_counts[annotation["class"]] += 1

        location = annotation.get("location_camera", [])
        if len(location) == 3:
            depths.append(float(location[2]))

print("=" * 50)
print("KITTI DATASET INSPECTION")
print("=" * 50)

print(f"Total matched frames: {len(records)}")
print(f"Frames with at least one supported vehicle: "
      f"{sum(count for n, count in objects_per_frame.items() if n > 0)}")

print("\nClass distribution:")
for class_name, count in sorted(class_counts.items()):
    print(f"  {class_name}: {count}")

print("\nObjects per frame:")
for count, frames in sorted(objects_per_frame.items()):
    print(f"  {count} vehicle(s): {frames} frame(s)")

if depths:
    print("\nAnnotated vehicle depth (camera Z):")
    print(f"  Minimum: {min(depths):.2f} m")
    print(f"  Maximum: {max(depths):.2f} m")
    print(f"  Mean:    {sum(depths) / len(depths):.2f} m")

print("\nInspection completed.")

