
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from train_lidar_detector import (
    LidarVehicleDataset,
    PointClassifier,
    MANIFEST_PATH,
    SEED,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = Path("outputs/lidar_point_classifier.pth")
BATCH_SIZE = 16
THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def calculate_metrics(y_true, y_pred):
    """Calculate point-level classification metrics."""

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    specificity = tn / max(tn + fp, 1)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def main():
    print("=" * 60)
    print("LiDAR VEHICLE CLASSIFIER — MODEL EVALUATION")
    print("=" * 60)

    # Check required files.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    # Reproduce the training validation split.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = LidarVehicleDataset(MANIFEST_PATH)

    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(SEED)

    _, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load the trained model.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # Run inference without updating model weights.
    with torch.no_grad():
        for features, labels in validation_loader:
            features = features.to(DEVICE)

            logits = model(features)
            probabilities = torch.sigmoid(logits)

            predictions = probabilities >= THRESHOLD

            all_labels.append(labels.cpu().numpy().reshape(-1))
            all_predictions.append(
                predictions.cpu().numpy().astype(np.int32).reshape(-1)
            )
            all_probabilities.append(
                probabilities.cpu().numpy().reshape(-1)
            )

    y_true = np.concatenate(all_labels).astype(np.int32)
    y_pred = np.concatenate(all_predictions).astype(np.int32)
    probabilities = np.concatenate(all_probabilities)

    metrics = calculate_metrics(y_true, y_pred)

    # Display results.
    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Total validation points: {len(y_true):,}")
    print(f"Decision threshold: {THRESHOLD:.2f}")

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-score:    {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

    print("\n--- Confusion Matrix ---")
    print("Rows = actual class; columns = predicted class")
    print()
    print("                   Predicted")
    print("                 Background  Vehicle")
    print(
        f"Actual Background {metrics['TN']:10,}"
        f" {metrics['FP']:8,}"
    )
    print(
        f"Actual Vehicle    {metrics['FN']:10,}"
        f" {metrics['TP']:8,}"
    )

    print("\n--- Error Analysis ---")
    print(f"False positives: {metrics['FP']:,}")
    print(f"Missed vehicle points: {metrics['FN']:,}")
    print(f"Mean vehicle probability: {probabilities.mean():.4f}")

    # Save the metrics for the project report.
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "model_evaluation.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("LiDAR Vehicle Classifier Evaluation\n")
        file.write("=" * 40 + "\n")
        file.write(f"Validation points: {len(y_true):,}\n")
        file.write(f"Threshold: {THRESHOLD:.2f}\n\n")

        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
        ):
            file.write(f"{name}: {metrics[name]:.4f}\n")

        file.write("\nConfusion Matrix Counts\n")
        file.write(f"True negatives: {metrics['TN']}\n")
        file.write(f"False positives: {metrics['FP']}\n")
        file.write(f"False negatives: {metrics['FN']}\n")
        file.write(f"True positives: {metrics['TP']}\n")

    print(f"\nEvaluation report saved to: {report_path.resolve()}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

