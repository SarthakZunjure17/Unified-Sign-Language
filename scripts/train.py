#!/usr/bin/env python3
"""
scripts/train.py
----------------
Phase 3 Model Training Pipeline for Unified Sign Language Interpreter.

Features:
- Hardware detection (TF version, Python, GPU/CPU).
- Data leakage verification across group_id.
- Streaming tf.data pipeline reading from CSV manifests (no full-RAM loading).
- EvilPort2 CNN topology modernized for TensorFlow 2.x with Rescaling(1./255).
- SGD optimizer, categorical crossentropy, EarlyStopping, ModelCheckpoint.
- Training history logging to results/training_history.json.
- Checkpointing and resume capability.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

def detect_hardware():
    """Detects and reports available compute hardware."""
    print("=" * 60)
    print(" HARDWARE & ENVIRONMENT DETECTION")
    print("=" * 60)
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"TensorFlow Version:  {tf.__version__}")
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU Availability:    AVAILABLE ({len(gpus)} device(s))")
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get("device_name", "NVIDIA GPU")
            except Exception:
                name = "GPU Device"
            print(f"  GPU #{i}:          {name} ({gpu.name})")
    else:
        print("GPU Availability:    NOT DETECTED (Using CPU)")
        cpu_count = os.cpu_count() or 1
        print(f"CPU Fallback:        ACTIVE ({cpu_count} logical cores available)")
    print("=" * 60 + "\n")

def check_data_leakage(train_csv: Path, val_csv: Path, test_csv: Path):
    """Verifies that no group_id leaks between train, val, and test splits."""
    print("Verifying data leakage across group splits...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    # Each group is uniquely identified by (class_id, group_id)
    train_groups = set(zip(df_train["class_id"], df_train["group_id"]))
    val_groups = set(zip(df_val["class_id"], df_val["group_id"]))
    test_groups = set(zip(df_test["class_id"], df_test["group_id"]))
    
    leak_train_val = train_groups & val_groups
    leak_train_test = train_groups & test_groups
    leak_val_test = val_groups & test_groups
    
    if leak_train_val:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_train_val)} groups overlap between Train and Val!")
    if leak_train_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_train_test)} groups overlap between Train and Test!")
    if leak_val_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_val_test)} groups overlap between Val and Test!")
        
    print(f"  [x] Train groups:      {len(train_groups)}")
    print(f"  [x] Validation groups: {len(val_groups)}")
    print(f"  [x] Test groups:       {len(test_groups)}")
    print("  [x] Zero overlap confirmed between all splits!\n")

def build_cnn_model(num_classes: int = 44) -> tf.keras.Model:
    """
    Reproduces the exact EvilPort2 CNN architecture modernized for TF 2.x:
    Input: (50, 50, 1) -> Rescaling(1./255)
    Conv2D(16, 2x2, relu, same) -> MaxPool(2x2, stride 2, same)
    Conv2D(32, 3x3, relu, same) -> MaxPool(3x3, stride 3, same)
    Conv2D(64, 5x5, relu, same) -> MaxPool(5x5, stride 5, same)
    Flatten -> Dense(128, relu) -> Dropout(0.2) -> Dense(num_classes, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 1), name="input_image"),
        tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling"),
        
        # Conv Block 1
        tf.keras.layers.Conv2D(16, (2, 2), padding="same", activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="maxpool_1"),
        
        # Conv Block 2
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same", name="maxpool_2"),
        
        # Conv Block 3
        tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same", name="maxpool_3"),
        
        # Classification Head
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
        tf.keras.layers.Dropout(0.2, name="dropout_0.2"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_softmax")
    ], name="EvilPort2_Modernized_CNN")
    
    return model

def create_tf_dataset(csv_path: Path, project_root: Path, batch_size: int = 128, is_training: bool = True) -> tf.data.Dataset:
    """
    Creates an efficient streaming tf.data pipeline reading from the CSV manifest.
    Does NOT load full images into RAM at once.
    """
    df = pd.read_csv(csv_path)
    # Build absolute paths
    paths = [str(project_root / p) for p in df["image_path"].values]
    labels = df["class_id"].values.astype(np.int32)
    
    def parse_image_and_label(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=1)
        img = tf.image.resize(img, [50, 50])
        img = tf.cast(img, tf.float32)
        one_hot = tf.one_hot(label, depth=44)
        return img, one_hot

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration=True)
        
    ds = ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

class CustomHistoryLogger(tf.keras.callbacks.Callback):
    """Logs epoch metrics and learning rate to results/training_history.json."""
    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self.history_records = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get learning rate
        lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer, "learning_rate") else 0.01
        record = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "learning_rate": lr
        }
        self.history_records.append(record)
        
        # Save incrementally
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history_records, f, indent=2)
            
        print(f"\n--- Epoch {epoch+1} Summary: "
              f"Loss: {record['loss']:.4f} | Acc: {record['accuracy']*100:.2f}% | "
              f"Val Loss: {record['val_loss']:.4f} | Val Acc: {record['val_accuracy']*100:.2f}% ---")

def main():
    parser = argparse.ArgumentParser(description="Train Modernized Sign Language CNN")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD Learning rate (default: 0.01)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    detect_hardware()
    
    metadata_dir = project_root / "metadata"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = metadata_dir / "train.csv"
    val_csv = metadata_dir / "validation.csv"
    test_csv = metadata_dir / "test.csv"
    
    # 1. Leakage check
    check_data_leakage(train_csv, val_csv, test_csv)
    
    # 2. Build or resume model
    best_model_path = models_dir / "sign_language_cnn_best.keras"
    final_model_path = models_dir / "sign_language_cnn.keras"
    
    if args.resume and best_model_path.exists():
        print(f"Resuming from existing checkpoint: {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
    else:
        print("Constructing modernized EvilPort2 CNN topology...")
        model = build_cnn_model(num_classes=44)
        
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # 3. Create streaming tf.data datasets
    print(f"\nBuilding streaming tf.data pipeline (batch size={args.batch_size})...")
    train_ds = create_tf_dataset(train_csv, project_root, batch_size=args.batch_size, is_training=True)
    val_ds = create_tf_dataset(val_csv, project_root, batch_size=args.batch_size, is_training=False)
    
    # 4. Configure Callbacks
    history_json_path = results_dir / "training_history.json"
    history_logger = CustomHistoryLogger(history_json_path)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, model_checkpoint, history_logger]
    
    # 5. Execute Training
    print("\n" + "=" * 60)
    print(f" STARTING TRAINING (Up to {args.epochs} epochs)")
    print("=" * 60)
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes).")
    
    # Save final model
    print(f"Saving final model to {final_model_path}...")
    model.save(str(final_model_path))
    
    # Record best validation metrics
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [0.0]))
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
