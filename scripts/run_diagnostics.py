#!/usr/bin/env python3
"""
scripts/run_diagnostics.py
--------------------------
Automated diagnostic pipeline for Phase 3.5:
- Data validation checks (leakage, class count, class 42 = "Fuck")
- Experiment B (Adam)
- Experiment C (Original Configuration)
- Experiment D (Modernized CNN with BatchNorm)
- Generates all training histories, evaluation reports, confusion matrices, and plots
- Produces results/diagnostic_comparison.csv and results/diagnostic_comparison.txt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def perform_data_validation(project_root: Path):
    """Rigorous dataset and split validation."""
    print("=" * 70)
    print(" STEP 8 - CRITICAL DATA VALIDATION")
    print("=" * 70)
    metadata_dir = project_root / "metadata"
    train_df = pd.read_csv(metadata_dir / "train.csv")
    val_df = pd.read_csv(metadata_dir / "validation.csv")
    test_df = pd.read_csv(metadata_dir / "test.csv")
    labels_df = pd.read_csv(metadata_dir / "labels.csv")
    with open(metadata_dir / "labels.json", "r", encoding="utf-8") as f:
        labels_json = json.load(f)

    # 1. Row counts
    print(f"Train sample count:      {len(train_df):,} (Expected: 73,920)")
    print(f"Validation sample count: {len(val_df):,} (Expected: 15,840)")
    print(f"Test sample count:       {len(test_df):,} (Expected: 15,840)")
    assert len(train_df) == 73920, f"Train count mismatch: {len(train_df)}"
    assert len(val_df) == 15840, f"Validation count mismatch: {len(val_df)}"
    assert len(test_df) == 15840, f"Test count mismatch: {len(test_df)}"

    # 2. Total classes & per-class count
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    class_counts = all_df["class_id"].value_counts()
    assert len(class_counts) == 44, f"Expected 44 classes, got {len(class_counts)}"
    for cid, cnt in class_counts.items():
        assert cnt == 2400, f"Class {cid} has {cnt} images, expected 2,400"
    print(f"Class count:             44 classes, exactly 2,400 images/class")

    # 3. Class 42 verification
    c42_row = labels_df[labels_df["class_id"] == 42]
    assert len(c42_row) == 1, "Class 42 missing from labels.csv"
    c42_label = c42_row.iloc[0]["label"]
    c42_entry = next((item for item in labels_json if item.get("class_id") == 42), None)
    assert c42_entry is not None and c42_entry.get("label") == "Fuck", "Class 42 not 'Fuck' in labels.json"
    print(f"Class 42 Label Check:    CONFIRMED -> '{c42_label}'")

    # 4. Group leakage check (class-aware original/flipped pair groups)
    train_groups = set(zip(train_df["class_id"], train_df["group_id"]))
    val_groups = set(zip(val_df["class_id"], val_df["group_id"]))
    test_groups = set(zip(test_df["class_id"], test_df["group_id"]))

    tv_overlap = train_groups.intersection(val_groups)
    tt_overlap = train_groups.intersection(test_groups)
    vt_overlap = val_groups.intersection(test_groups)

    print(f"Train INTERSECT Val:     {len(tv_overlap)} shared groups")
    print(f"Train INTERSECT Test:    {len(tt_overlap)} shared groups")
    print(f"Validation INTERSECT Test: {len(vt_overlap)} shared groups")

    assert len(tv_overlap) == 0, f"Data leakage between Train and Val: {len(tv_overlap)} groups"
    assert len(tt_overlap) == 0, f"Data leakage between Train and Test: {len(tt_overlap)} groups"
    assert len(vt_overlap) == 0, f"Data leakage between Val and Test: {len(vt_overlap)} groups"

    print("Result:                  ALL DATA VALIDATION CHECKS PASSED (ZERO LEAKAGE)")
    print("=" * 70 + "\n")

def make_dataset(csv_path: Path, project_root: Path, batch_size: int, is_training: bool = False, rescale: bool = True):
    df = pd.read_csv(csv_path)
    paths = [str(project_root / p) for p in df["image_path"].values]
    labels = df["class_id"].values.astype(np.int32)
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=44)

    def parse_fn(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=1)
        img = tf.image.resize(img, [50, 50])
        img = tf.cast(img, tf.float32)
        if rescale:
            img = img / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_one_hot))
    if is_training:
        ds = ds.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_baseline_cnn(rescale_in_model: bool = False):
    """Recreates the exact EvilPort2 modernized topology."""
    inputs = tf.keras.Input(shape=(50, 50, 1), name="input")
    x = inputs
    if rescale_in_model:
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
    x = tf.keras.layers.Conv2D(16, (2, 2), padding="same", activation="relu", name="conv2d_1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="maxpool_1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same", name="maxpool_2")(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2d_3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same", name="maxpool_3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_0.2")(x)
    outputs = tf.keras.layers.Dense(44, activation="softmax", name="output_softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

def build_modern_cnn():
    """Builds a modestly modernized CNN with BatchNormalization."""
    inputs = tf.keras.Input(shape=(50, 50, 1), name="input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv_b1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv_b2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.ReLU(name="relu_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv_b3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.ReLU(name="relu_3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)

    # Classifier
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, name="dense_256")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_dense")(x)
    x = tf.keras.layers.ReLU(name="relu_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_0.3")(x)
    outputs = tf.keras.layers.Dense(44, activation="softmax", name="output_softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="modern_batchnorm_cnn")

def plot_training_curves(history_records: list, output_dir: Path):
    epochs = [r["epoch"] for r in history_records]
    acc = [r["accuracy"] * 100 for r in history_records]
    val_acc = [r["val_accuracy"] * 100 for r in history_records]
    loss = [r["loss"] for r in history_records]
    val_loss = [r["val_loss"] for r in history_records]
    
    # Accuracy
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, acc, "o-", color="#1f77b4", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, "s-", color="#ff7f0e", label="Validation Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png")
    plt.close()

    # Loss
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, loss, "o-", color="#d62728", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, "s-", color="#2ca02c", label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_labels, output_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=16, pad=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)
    plt.xlabel("Predicted Gesture Label", fontsize=14, labelpad=10)
    plt.ylabel("Ground Truth Gesture Label", fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

class CustomHistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.history_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        rec = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "learning_rate": float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer, "learning_rate") else 0.0
        }
        self.history_records.append(rec)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history_records, f, indent=2)

def train_and_evaluate_experiment(
    exp_name: str,
    model_func,
    optimizer,
    batch_size: int,
    epochs: int,
    rescale_data: bool,
    rescale_in_model: bool,
    model_out_path: Path,
    results_dir: Path,
    project_root: Path
):
    print("\n" + "=" * 70)
    print(f" RUNNING EXPERIMENT: {exp_name}")
    print(f" Optimizer: {type(optimizer).__name__} | Batch: {batch_size} | Rescale: {rescale_data or rescale_in_model}")
    print(f" Target Model:   {model_out_path}")
    print(f" Target Results: {results_dir}")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = project_root / "metadata"
    train_csv = metadata_dir / "train.csv"
    val_csv = metadata_dir / "validation.csv"
    test_csv = metadata_dir / "test.csv"
    labels_df = pd.read_csv(metadata_dir / "labels.csv")
    class_labels = labels_df["label"].tolist()

    # Build datasets
    train_ds = make_dataset(train_csv, project_root, batch_size, is_training=True, rescale=rescale_data)
    val_ds = make_dataset(val_csv, project_root, batch_size, is_training=False, rescale=rescale_data)
    test_ds = make_dataset(test_csv, project_root, batch_size, is_training=False, rescale=rescale_data)

    # Build model
    if model_func == build_baseline_cnn:
        model = model_func(rescale_in_model=rescale_in_model)
    else:
        model = model_func()

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history_log_path = results_dir / "training_history.json"
    history_cb = CustomHistoryLogger(history_log_path)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_out_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[history_cb, ckpt_cb, early_cb],
        verbose=1
    )
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.2f} min).")

    # Ensure best model is saved
    model.save(str(model_out_path))

    # Evaluate on test set
    test_df = pd.read_csv(test_csv)
    y_true = test_df["class_id"].values.astype(np.int32)
    print(f"Evaluating {exp_name} on {len(y_true)} test samples...")
    raw_preds = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(raw_preds, axis=1)

    test_acc = float(np.mean(y_pred == y_true))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
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
    df_per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)

    # Classification report
    clf_report = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"DIAGNOSTIC REPORT: {exp_name}\n")
        f.write(f"Model: {model_out_path.name}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Macro F1: {f1_macro*100:.2f}%\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report)

    # Metrics JSON
    metrics = {
        "experiment": exp_name,
        "test_accuracy": round(test_acc, 4),
        "test_samples": int(len(y_true)),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "training_time_seconds": round(train_time, 2)
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_training_curves(history_cb.history_records, results_dir)
    plot_confusion_matrix(y_true, y_pred, class_labels, results_dir / "confusion_matrix.png", f"{exp_name} Confusion Matrix")

    print(f"Results for {exp_name}:")
    print(f"  Test Accuracy:     {test_acc*100:.2f}%")
    print(f"  Macro F1:          {f1_macro*100:.2f}%")
    print(f"  Weighted F1:       {f1_weighted*100:.2f}%")
    
    return {
        "model": exp_name,
        "normalization": "Rescaling [0, 1]" if (rescale_data or rescale_in_model) else "None [0, 255]",
        "optimizer": type(optimizer).__name__,
        "batch": batch_size,
        "test_accuracy": f"{test_acc*100:.2f}%",
        "macro_f1": f"{f1_macro*100:.2f}%",
        "weighted_f1": f"{f1_weighted*100:.2f}%"
    }

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3.5 Diagnostics")
    parser.add_argument("--skip-validation", action="store_true", help="Skip initial data validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    if not args.skip_validation:
        perform_data_validation(project_root)

    comparison_rows = []

    # 1. Original pretrained model from results/original_model_benchmark/metrics.json
    orig_metrics_path = results_dir / "original_model_benchmark" / "metrics.json"
    if orig_metrics_path.exists():
        with open(orig_metrics_path, "r", encoding="utf-8") as f:
            om = json.load(f)
            comparison_rows.append({
                "model": "Original Pretrained (cnn_model_keras2.h5)",
                "normalization": "Raw [0, 255]",
                "optimizer": "SGD (Historical)",
                "batch": 500,
                "test_accuracy": f"{om['test_accuracy']*100:.2f}%",
                "macro_f1": f"{om['macro_f1']*100:.2f}%",
                "weighted_f1": f"{om['weighted_f1']*100:.2f}%"
            })

    # 2. Phase 3 baseline model
    base_metrics_path = results_dir / "metrics.json"
    if base_metrics_path.exists():
        with open(base_metrics_path, "r", encoding="utf-8") as f:
            bm = json.load(f)
            comparison_rows.append({
                "model": "Current Baseline (Experiment A)",
                "normalization": "Rescaling [0, 1]",
                "optimizer": "SGD (lr=0.01)",
                "batch": 128,
                "test_accuracy": f"{bm['test_accuracy']*100:.2f}%",
                "macro_f1": f"{bm['macro_f1']*100:.2f}%",
                "weighted_f1": f"{bm['weighted_f1']*100:.2f}%"
            })

    # 3. Experiment B: Adam
    adam_res = train_and_evaluate_experiment(
        exp_name="Experiment B - Adam Optimizer",
        model_func=build_baseline_cnn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        batch_size=128,
        epochs=20,
        rescale_data=False,
        rescale_in_model=True,
        model_out_path=models_dir / "diagnostic_adam.keras",
        results_dir=results_dir / "diagnostic_adam",
        project_root=project_root
    )
    comparison_rows.append(adam_res)

    # 4. Experiment C: Original Training Configuration (Raw [0, 255] pixels, SGD lr=0.01, Batch=500)
    orig_cfg_res = train_and_evaluate_experiment(
        exp_name="Experiment C - Original Config (Raw Pixels, SGD, Batch 500)",
        model_func=build_baseline_cnn,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
        batch_size=500,
        epochs=20,
        rescale_data=False,
        rescale_in_model=False,
        model_out_path=models_dir / "diagnostic_original_config.keras",
        results_dir=results_dir / "diagnostic_original_config",
        project_root=project_root
    )
    comparison_rows.append(orig_cfg_res)

    # 5. Experiment D: Modernized CNN with BatchNormalization
    modern_res = train_and_evaluate_experiment(
        exp_name="Experiment D - Modern CNN (BatchNorm + Adam)",
        model_func=build_modern_cnn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        batch_size=128,
        epochs=20,
        rescale_data=False,
        rescale_in_model=False,  # rescale is built into build_modern_cnn
        model_out_path=models_dir / "diagnostic_modern_cnn.keras",
        results_dir=results_dir / "diagnostic_modern_cnn",
        project_root=project_root
    )
    comparison_rows.append(modern_res)

    # Generate Comparison Table CSV & TXT
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = results_dir / "diagnostic_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved comparison CSV to: {comp_csv}")

    comp_txt = results_dir / "diagnostic_comparison.txt"
    with open(comp_txt, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(" PHASE 3.5 DIAGNOSTIC MODEL COMPARISON\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Model':<40} | {'Norm':<16} | {'Optimizer':<18} | {'Batch':<6} | {'Test Acc':<9} | {'Macro F1':<9} | {'Weighted F1':<11}\n")
        f.write("-" * 95 + "\n")
        for r in comparison_rows:
            f.write(f"{r['model']:<40} | {r['normalization']:<16} | {r['optimizer']:<18} | {str(r['batch']):<6} | {r['test_accuracy']:<9} | {r['macro_f1']:<9} | {r['weighted_f1']:<11}\n")
        f.write("=" * 95 + "\n")
    print(f"Saved comparison TXT to: {comp_txt}")

    print("\n" + "=" * 95)
    print(" ALL DIAGNOSTIC EXPERIMENTS COMPLETE")
    print("=" * 95)

if __name__ == "__main__":
    main()
