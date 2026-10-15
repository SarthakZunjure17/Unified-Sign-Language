#!/usr/bin/env python3
"""
scripts/evaluate_original_model.py
----------------------------------
Reconstructs the original EvilPort2 model architecture and loads weights
directly from the untouched cnn_model_keras2.h5 file, then evaluates on
the exact 15,840-image leakage-safe test set (metadata/test.csv).
"""

import os
import sys
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def build_and_load_original_model(orig_h5_path: Path):
    """Builds original EvilPort2 architecture and loads weights from HDF5."""
    inputs = tf.keras.Input(shape=(50, 50, 1), name="input")
    x = tf.keras.layers.Conv2D(16, (2, 2), padding="valid", activation="relu", name="conv2d_1")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pooling2d_1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="valid", activation="relu", name="conv2d_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same", name="max_pooling2d_2")(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding="valid", activation="relu", name="conv2d_3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same", name="max_pooling2d_3")(x)
    x = tf.keras.layers.Flatten(name="flatten_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(44, activation="softmax", name="dense_2")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="original_evilport2_model")

    with h5py.File(orig_h5_path, "r") as f:
        w_group = f["model_weights"]
        model.get_layer("conv2d_1").set_weights([
            w_group["conv2d_1"]["conv2d_1"]["kernel:0"][:],
            w_group["conv2d_1"]["conv2d_1"]["bias:0"][:]
        ])
        model.get_layer("conv2d_2").set_weights([
            w_group["conv2d_2"]["conv2d_2"]["kernel:0"][:],
            w_group["conv2d_2"]["conv2d_2"]["bias:0"][:]
        ])
        model.get_layer("conv2d_3").set_weights([
            w_group["conv2d_3"]["conv2d_3"]["kernel:0"][:],
            w_group["conv2d_3"]["conv2d_3"]["bias:0"][:]
        ])
        model.get_layer("dense_1").set_weights([
            w_group["dense_1"]["dense_1"]["kernel:0"][:],
            w_group["dense_1"]["dense_1"]["bias:0"][:]
        ])
        model.get_layer("dense_2").set_weights([
            w_group["dense_2"]["dense_2"]["kernel:0"][:],
            w_group["dense_2"]["dense_2"]["bias:0"][:]
        ])

    return model

def load_test_dataset(test_csv: Path, project_root: Path, batch_size: int = 128, rescale: bool = False):
    df = pd.read_csv(test_csv)
    paths = [str(project_root / p) for p in df["image_path"].values]
    labels = df["class_id"].values.astype(np.int32)
    
    def parse_img(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=1)
        img = tf.image.resize(img, [50, 50])
        img = tf.cast(img, tf.float32)
        if rescale:
            img = img / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds, labels, df

def plot_confusion_matrix(y_true, y_pred, class_labels, output_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Original EvilPort2 Model Confusion Matrix (44 Classes)", fontsize=16, pad=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)
    plt.xlabel("Predicted Gesture Label", fontsize=14, labelpad=10)
    plt.ylabel("Ground Truth Gesture Label", fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    project_root = Path(__file__).resolve().parent.parent
    orig_h5_path = project_root.parent / "Sign-Language-master" / "Sign-Language-master" / "cnn_model_keras2.h5"
    metadata_dir = project_root / "metadata"
    output_dir = project_root / "results" / "original_model_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading weights from untouched original model: {orig_h5_path}")
    model = build_and_load_original_model(orig_h5_path)
    
    labels_csv = metadata_dir / "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    class_labels = labels_df["label"].tolist()
    test_csv = metadata_dir / "test.csv"

    # Evaluate with raw pixel values (as in original load_images.py / recognize_gesture.py)
    print("Evaluating original model with raw [0, 255] pixels on 15,840 test samples...")
    test_ds_raw, y_true, _ = load_test_dataset(test_csv, project_root, batch_size=128, rescale=False)
    preds_raw = model.predict(test_ds_raw, verbose=1)
    y_pred_raw = np.argmax(preds_raw, axis=1)
    acc_raw = float(np.mean(y_pred_raw == y_true))

    # Also evaluate with rescaled [0, 1] pixels for completeness
    print("Evaluating original model with rescaled [0, 1] pixels on 15,840 test samples...")
    test_ds_norm, _, _ = load_test_dataset(test_csv, project_root, batch_size=128, rescale=True)
    preds_norm = model.predict(test_ds_norm, verbose=1)
    y_pred_norm = np.argmax(preds_norm, axis=1)
    acc_norm = float(np.mean(y_pred_norm == y_true))

    print(f"Original Model Test Accuracy (Raw [0, 255]):      {acc_raw*100:.2f}%")
    print(f"Original Model Test Accuracy (Rescaled [0, 1]):    {acc_norm*100:.2f}%")

    # Select the better/intended representation
    if acc_raw >= acc_norm:
        y_pred = y_pred_raw
        chosen_acc = acc_raw
        best_mode = "raw [0, 255]"
    else:
        y_pred = y_pred_norm
        chosen_acc = acc_norm
        best_mode = "rescaled [0, 1]"

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per class metrics
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
    per_class_csv_path = output_dir / "per_class_metrics.csv"
    df_per_class.to_csv(per_class_csv_path, index=False)

    # Classification report
    clf_report = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ORIGINAL EVILPORT2 PRETRAINED MODEL - CLASSIFICATION REPORT\n")
        f.write(f"Source: Sign-Language-master/Sign-Language-master/cnn_model_keras2.h5\n")
        f.write(f"Preprocessing: {best_mode}\n")
        f.write(f"Raw Accuracy: {acc_raw*100:.2f}%, Rescaled Accuracy: {acc_norm*100:.2f}%\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report)

    # Metrics JSON
    metrics = {
        "model": "cnn_model_keras2.h5 (Original EvilPort2 Pretrained)",
        "preprocessing_evaluated": {
            "raw_0_255_accuracy": round(acc_raw, 4),
            "rescaled_0_1_accuracy": round(acc_norm, 4),
            "best_mode": best_mode
        },
        "test_accuracy": round(chosen_acc, 4),
        "test_samples": int(len(y_true)),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4)
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_labels, cm_path)

    print(f"\nOriginal Model Benchmark complete.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
