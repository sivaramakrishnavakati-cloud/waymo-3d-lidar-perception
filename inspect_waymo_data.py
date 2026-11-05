import numpy as np
import pandas as pd

# Load actual Waymo LiDAR frame and ground-truth labels
points = np.load("data/waymo/processed/waymo_frame_000.npy")
labels = pd.read_csv("data/waymo/processed/waymo_labels_000.csv")

print("WAYMO DATA INSPECTION")
print("-" * 40)

print("Point cloud shape:", points.shape)
print("Number of LiDAR points:", len(points))

print("\nPoint cloud coordinate ranges:")
print("X:", points[:, 0].min(), "to", points[:, 0].max())
print("Y:", points[:, 1].min(), "to", points[:, 1].max())
print("Z:", points[:, 2].min(), "to", points[:, 2].max())

print("\nGround-truth labels:", len(labels))
print("Label columns:", labels.columns.tolist())

print("\nFirst five labels:")
print(labels.head())

print("\nData inspection completed!")