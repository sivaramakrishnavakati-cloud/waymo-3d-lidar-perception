
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "kitti_temp"
DEST = PROJECT / "data" / "kitti"

# Source and destination folders
folders = {
    SOURCE / "velodyne" / "training" / "velodyne":
        DEST / "training" / "velodyne",

    SOURCE / "velodyne" / "testing" / "velodyne":
        DEST / "testing" / "velodyne",

    SOURCE / "calib" / "training" / "calib":
        DEST / "training" / "calib",

    SOURCE / "calib" / "testing" / "calib":
        DEST / "testing" / "calib",

    SOURCE / "label_2" / "training" / "label_2":
        DEST / "training" / "label_2",
}

for src, dst in folders.items():
    if not src.is_dir():
        print(f"SKIPPED - source folder not found: {src}")
        continue

    files = [p for p in src.iterdir() if p.is_file()]
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0

    for file in files:
        target = dst / file.name

        if target.exists():
            print(f"SKIPPED - destination already exists: {target.name}")
            continue

        shutil.copy2(file, target)
        copied += 1

    print(f"Copied {copied} files: {src} -> {dst}")

print("\nDataset organization finished.")
print("Original files remain in kitti_temp.")

