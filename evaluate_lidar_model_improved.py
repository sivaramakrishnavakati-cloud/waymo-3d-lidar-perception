from pathlib import Path
import csv
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
OUTPUT_DIR = Path("outputs")
BATCH_SIZE = 16
BASELINE_THRESHOLD = 0.50
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 2)
# Useful for selecting a less noisy threshold without completely sacrificing recall.
MIN_RECALL_FOR_PRECISION_PICK = 0.70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate point-level binary classification metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def main() -> None:
    print("=" * 68)
    print("LiDAR VEHICLE CLASSIFIER — THRESHOLD EVALUATION")
    print("=" * 68)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH.resolve()}")
    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(f"Dataset manifest not found: {Path(MANIFEST_PATH).resolve()}")

    # Reproduce the same deterministic validation split used by training/evaluation.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dataset = LidarVehicleDataset(MANIFEST_PATH)
    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(SEED)
    _, validation_set = random_split(
        dataset, [training_size, validation_size], generator=generator
    )
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = PointClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_all = []
    probabilities_all = []
    with torch.no_grad():
        for features, labels in validation_loader:
            logits = model(features.to(DEVICE))
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probabilities_all.append(probs)
            labels_all.append(labels.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(labels_all).astype(np.int32)
    probabilities = np.concatenate(probabilities_all)

    results = []
    for threshold in THRESHOLDS:
        y_pred = (probabilities >= threshold).astype(np.int32)
        row = {"threshold": float(threshold), **calculate_metrics(y_true, y_pred)}
        results.append(row)

    baseline = min(results, key=lambda row: abs(row["threshold"] - BASELINE_THRESHOLD))
    best_f1 = max(results, key=lambda row: row["f1"])
    eligible = [r for r in results if r["recall"] >= MIN_RECALL_FOR_PRECISION_PICK]
    best_precision = max(eligible, key=lambda row: row["precision"]) if eligible else best_f1

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation points: {len(y_true):,}")
    print("\n--- Threshold comparison ---")
    print("Threshold | Accuracy | Precision | Recall | F1-score | Specificity | False positives | Missed vehicles")
    print("-" * 112)
    for r in results:
        marker = "  <-- baseline" if abs(r["threshold"] - BASELINE_THRESHOLD) < 1e-9 else ""
        print(
            f"{r['threshold']:9.2f} | {r['accuracy']:8.4f} | {r['precision']:9.4f} | "
            f"{r['recall']:6.4f} | {r['f1']:8.4f} | {r['specificity']:11.4f} | "
            f"{r['FP']:15,} | {r['FN']:15,}{marker}"
        )

    print("\n--- Recommendations from this validation split ---")
    print(
        f"Best F1 threshold: {best_f1['threshold']:.2f} "
        f"(precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}, "
        f"F1={best_f1['f1']:.4f}, false positives={best_f1['FP']:,})"
    )
    print(
        f"Best precision with recall >= {MIN_RECALL_FOR_PRECISION_PICK:.2f}: "
        f"threshold={best_precision['threshold']:.2f} "
        f"(precision={best_precision['precision']:.4f}, recall={best_precision['recall']:.4f}, "
        f"F1={best_precision['f1']:.4f}, false positives={best_precision['FP']:,})"
    )
    print(
        f"Baseline threshold {BASELINE_THRESHOLD:.2f}: precision={baseline['precision']:.4f}, "
        f"recall={baseline['recall']:.4f}, F1={baseline['f1']:.4f}, "
        f"false positives={baseline['FP']:,}, missed vehicle points={baseline['FN']:,}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "threshold_comparison.csv"
    fields = ["threshold", "accuracy", "precision", "recall", "f1", "specificity", "TP", "TN", "FP", "FN"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r[key] for key in fields})

    report_path = OUTPUT_DIR / "threshold_evaluation.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("LiDAR Vehicle Classifier — Threshold Evaluation\n")
        f.write("=" * 48 + "\n")
        f.write(f"Model: {MODEL_PATH}\nValidation points: {len(y_true):,}\n")
        f.write("Note: threshold selection on this validation split is diagnostic; confirm the final choice on a separate held-out test set.\n\n")
        f.write("Threshold comparison\n")
        for r in results:
            f.write(
                f"threshold={r['threshold']:.2f}, accuracy={r['accuracy']:.4f}, "
                f"precision={r['precision']:.4f}, recall={r['recall']:.4f}, "
                f"f1={r['f1']:.4f}, specificity={r['specificity']:.4f}, "
                f"FP={r['FP']}, FN={r['FN']}\n"
            )
        f.write("\nSuggested by best F1 on this validation split\n")
        f.write(f"threshold={best_f1['threshold']:.2f}, precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}, f1={best_f1['f1']:.4f}\n")
        f.write("\nSuggested by best precision while meeting recall floor\n")
        f.write(f"recall_floor={MIN_RECALL_FOR_PRECISION_PICK:.2f}, threshold={best_precision['threshold']:.2f}, precision={best_precision['precision']:.4f}, recall={best_precision['recall']:.4f}, f1={best_precision['f1']:.4f}\n")

    print(f"\nCSV saved: {csv_path.resolve()}")
    print(f"Report saved: {report_path.resolve()}")
    print("\nEvaluation complete. This script does not change or overwrite your trained model.")


if __name__ == "__main__":
    main()
