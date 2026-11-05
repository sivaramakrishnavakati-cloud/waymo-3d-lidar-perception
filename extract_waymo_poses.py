import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from waymo_open_dataset import dataset_pb2

# Original Waymo dataset
TFRECORD_PATH = Path(
    "data/waymo/Training/waymo_segment_000.tfrecord.tfrecord"
)

# Output directory
OUTPUT_DIR = Path("data/waymo/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pose_records = []

dataset = tf.data.TFRecordDataset(
    str(TFRECORD_PATH),
    compression_type=""
)

print("Extracting Waymo vehicle poses...")

for frame_idx, record in enumerate(dataset):

    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())

    pose = np.array(
        frame.pose.transform,
        dtype=np.float64
    ).reshape(4, 4)

    pose_records.append({
        "frame": frame_idx,
        "timestamp": frame.timestamp_micros,
        "pose": pose.flatten().tolist()
    })

    if (frame_idx + 1) % 10 == 0:
        print(f"Processed {frame_idx + 1} frames")

    if frame_idx >= 99:
        break

df = pd.DataFrame(pose_records)

output_file = OUTPUT_DIR / "waymo_poses.csv"
df.to_csv(output_file, index=False)

print("\nPose extraction completed!")
print("Frames extracted:", len(df))
print("Saved to:", output_file.resolve())
print("Pose matrix shape:", np.array(pose_records[0]["pose"]).reshape(4, 4).shape)