#!/usr/bin/env python3
"""
scripts/evaluate.py
-------------------
Comprehensive Evaluation and Analysis Utility for Unified Sign Language Interpreter.

Features:
- Evaluates best trained model on metadata/test.csv.
- Generates results/classification_report.txt and results/metrics.json.
- Generates results/per_class_metrics.csv.
- Generates high-resolution results/confusion_matrix.png.
- Generates results/training_accuracy.png and results/training_loss.png.
- Generates results/model_summary.txt.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def load_test_dataset(test_csv: Path, project_root: Path, batch_size: int = 128):
    """Creates a streaming tf.data dataset for test evaluation without shuffle."""
    df = pd.read_csv(test_csv)
    paths = [str(project_root / p) for p in df["image_path"].values]
    labels = df["class_id"].values.astype(np.int32)
    
    def parse_img(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=1)
        img = tf.image.resize(img, [50, 50])
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds, labels, df

def plot_training_curves(history_path: Path, results_dir: Path):
    """Plots training vs validation accuracy and loss curves."""
    if not history_path.exists():
        print(f"Warning: {history_path} not found, skipping training curves.")
        return
        
    with open(history_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    epochs = [r["epoch"] for r in records]
    acc = [r["accuracy"] * 100 for r in records]
    val_acc = [r["val_accuracy"] * 100 for r in records]
    loss = [r["loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    
    # Accuracy plot
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, acc, "o-", color="#1f77b4", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, "s-", color="#ff7f0e", label="Validation Accuracy", linewidth=2)
    plt.title("Sign Language CNN: Training vs Validation Accuracy", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    acc_path = results_dir / "training_accuracy.png"
    plt.savefig(acc_path)
    plt.close()
    print(f"Created {acc_path}")
    
    # Loss plot
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, loss, "o-", color="#d62728", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, "s-", color="#2ca02c", label="Validation Loss", linewidth=2)
    plt.title("Sign Language CNN: Training vs Validation Loss", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    loss_path = results_dir / "training_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"Created {loss_path}")

def plot_confusion_matrix(y_true, y_pred, class_labels, output_path: Path):
    """Generates a clear 44x44 confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(18, 16), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (44 Gesture Classes)", fontsize=18, pad=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)
    plt.xlabel("Predicted Gesture Label", fontsize=14, labelpad=10)
    plt.ylabel("Ground Truth Gesture Label", fontsize=14, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Created {output_path}")

def generate_model_summary(model, results_dir: Path, best_val_acc: float, test_acc: float, epochs: int = 20):
    """Outputs results/model_summary.txt."""
    summary_path = results_dir / "model_summary.txt"
    
    trainable_count = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = int(model.count_params() - trainable_count)
    
    lines = []
    lines.append("==================================================")
    lines.append(" UNIFIED SIGN LANGUAGE INTERPRETER - MODEL SUMMARY")
    lines.append("==================================================")
    lines.append(f"Input Shape:              (50, 50, 1)")
    lines.append(f"Output Classes:           44")
    lines.append(f"Total Parameters:         {model.count_params():,}")
    lines.append(f"Trainable Parameters:     {trainable_count:,}")
    lines.append(f"Non-Trainable Parameters: {non_trainable_count:,}")
    lines.append(f"Architecture:             Modernized EvilPort2 CNN [Conv(16,2x2)->Pool(2x2)->Conv(32,3x3)->Pool(3x3)->Conv(64,5x5)->Pool(5x5)->Dense(128)->Dropout(0.2)->Dense(44)]")
    lines.append(f"Optimizer:                SGD (momentum=0.0)")
    lines.append(f"Learning Rate:            0.01")
    lines.append(f"Batch Size:               128")
    lines.append(f"Epochs:                   {epochs}")
    lines.append(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    lines.append(f"Final Test Accuracy:      {test_acc*100:.2f}%")
    lines.append("-" * 50)
    lines.append("Detailed Layer Breakdown:")
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    lines.extend(stringlist)
    lines.append("==================================================")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Created {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Sign Language CNN")
    parser.add_argument("--model", type=str, default="", help="Path to saved model (.keras)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    metadata_dir = project_root / "metadata"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    
    # 1. Locate model
    if args.model:
        model_path = Path(args.model)
    else:
        best_candidate = models_dir / "sign_language_cnn_best.keras"
        final_candidate = models_dir / "sign_language_cnn.keras"
        if best_candidate.exists():
            model_path = best_candidate
        elif final_candidate.exists():
            model_path = final_candidate
        else:
            raise FileNotFoundError("No trained model found in models/. Run train.py first.")
            
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))
    
    # 2. Load labels
    labels_csv = metadata_dir / "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    class_labels = labels_df["label"].tolist()
    
    # 3. Load test data
    test_csv = metadata_dir / "test.csv"
    test_ds, y_true, test_df = load_test_dataset(test_csv, project_root, batch_size=args.batch_size)
    
    # 4. Perform Inference
    print(f"Running evaluation across {len(y_true)} test samples...")
    raw_preds = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(raw_preds, axis=1)
    
    # 5. Compute Metrics
    test_acc = float(np.mean(y_pred == y_true))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Generate per-class CSV
    per_class_rows = []
    for cid in range(len(class_labels)):
        per_class_rows.append({
            "class_id": cid,
            "label": class_labels[cid],
            "precision": round(float(prec_per_class[cid]), 4),
            "recall": round(float(rec_per_class[cid]), 4),
            "f1_score": round(float(f1_per_class[cid]), 4),
            "support": int(support_per_class[cid])
        })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv_path = results_dir / "per_class_metrics.csv"
    df_per_class.to_csv(per_class_csv_path, index=False)
    print(f"Created {per_class_csv_path}")
    
    # Generate classification_report.txt
    clf_report_text = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("UNIFIED SIGN LANGUAGE INTERPRETER - CLASSIFICATION REPORT\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report_text)
    print(f"Created {report_path}")
    
    # Generate metrics.json
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_samples": int(len(y_true)),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4)
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Created {metrics_path}")
    
    # Generate confusion_matrix.png
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_labels, cm_path)
    
    # Generate training curves
    history_json_path = results_dir / "training_history.json"
    plot_training_curves(history_json_path, results_dir)
    
    # Find best val accuracy and epoch count from history
    best_val_acc = test_acc
    epochs_count = 20
    if history_json_path.exists():
        with open(history_json_path, "r", encoding="utf-8") as f:
            h = json.load(f)
            best_val_acc = max([x.get("val_accuracy", 0.0) for x in h] or [test_acc])
            epochs_count = len(h)
            
    # Generate model_summary.txt
    generate_model_summary(model, results_dir, best_val_acc, test_acc, epochs=epochs_count)
    
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Macro F1 Score:        {f1_macro * 100:.2f}%")
    print(f"Weighted F1 Score:     {f1_weighted * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
