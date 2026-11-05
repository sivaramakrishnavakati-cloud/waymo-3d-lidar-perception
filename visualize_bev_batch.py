import csv
import json
from pathlib import Path

import torch
import matplotlib

# Use a non-interactive backend for batch processing.
matplotlib.use("Agg")

import visualize_bev_predictions as visualizer


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/bev_batch_results")

MAX_FRAMES = 20
START_INDEX = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BEV VEHICLE DETECTION - MULTI-FRAME INFERENCE")
    print("=" * 60)

    # Reuse the existing visualizer configuration.
    visualizer.OUTPUT_DIR = OUTPUT_DIR

    # Prevent a plot window from opening for every frame.
    visualizer.plt.show = lambda: None

    # Load dataset manifest.
    print("\nLoading KITTI manifest...")

    records = visualizer.load_manifest()

    print(f"Total manifest records: {len(records)}")

    # Select frames.
    selected_records = records[
        START_INDEX:START_INDEX + MAX_FRAMES
    ]

    # Load model only once.
    print("\nLoading trained model...")

    model = visualizer.load_model()

    results = []

    processed = 0
    skipped = 0

    # Process each frame.
    for index, record in enumerate(selected_records):

        frame_id = visualizer.get_frame_id(record)

        print("\n" + "-" * 50)
        print(f"Processing frame: {frame_id}")

        try:

            lidar_value = record.get("lidar_path")

            if not lidar_value:
                raise ValueError("Missing LiDAR path")

            # Load LiDAR points.
            points = visualizer.load_lidar_points(
                lidar_value
            )

            # Convert point cloud into BEV.
            bev = visualizer.make_bev(points)

            input_tensor = (
                torch.from_numpy(bev)
                .unsqueeze(0)
                .to(visualizer.DEVICE)
            )

            # Run inference.
            with torch.no_grad():

                prediction = model(input_tensor)[0]

            # Decode vehicle predictions.
            objectness, detections = (
                visualizer.decode_predictions(prediction)
            )

            # Generate and save visualization.
            visualizer.visualize(
                points,
                bev,
                objectness,
                detections,
                frame_id,
            )

            # Close figure to release memory.
            visualizer.plt.close("all")

            # Store frame results.
            results.append({
                "frame_id": frame_id,
                "points": len(points),
                "detections": len(detections),
                "max_objectness": float(objectness.max()),
                "status": "Success",
            })

            processed += 1

            print(f"Points: {len(points):,}")
            print(f"Detections: {len(detections)}")
            print(
                f"Maximum objectness: "
                f"{float(objectness.max()):.4f}"
            )

        except Exception as error:

            print(f"Skipped frame {frame_id}: {error}")

            results.append({
                "frame_id": frame_id,
                "points": 0,
                "detections": 0,
                "max_objectness": 0,
                "status": f"Failed: {error}",
            })

            skipped += 1

    # Save summary as CSV.
    summary_path = OUTPUT_DIR / "batch_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "points",
                "detections",
                "max_objectness",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # Save summary as JSON.
    json_path = OUTPUT_DIR / "batch_summary.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(results, file, indent=2)

    # Final report.
    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)

    print(f"Frames selected: {len(selected_records)}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped frames: {skipped}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"CSV summary: {summary_path}")
    print(f"JSON summary: {json_path}")

    print("\nInference demo completed successfully.")


if __name__ == "__main__":
    main()