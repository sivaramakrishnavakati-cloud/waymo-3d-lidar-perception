import pandas as pd
from pathlib import Path

folder = Path("data/waymo/processed/multi_frame")

for i in range(5):
    file = folder / f"frame_{i:04d}.csv"
    df = pd.read_csv(file)

    vehicles = df[df["type"] == 1]

    print(f"\nFrame {i}")
    print("Total objects:", len(df))
    print("Vehicles:", len(vehicles))
    print("Object IDs:")
    print(vehicles["object_id"].tolist())