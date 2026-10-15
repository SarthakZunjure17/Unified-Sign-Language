#!/usr/bin/env python3
"""
scripts/predict.py
------------------
Single-image and batch inference CLI utility for Unified Sign Language Interpreter.
Uses primary model (models/diagnostic_modern_cnn.keras) and metadata/labels.json.

Usage:
    python scripts/predict.py --image dataset/master/A/1.jpg
    python scripts/predict.py --image dataset/master/A/1.jpg --top-k 5
    python scripts/predict.py --verify-random 10
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.predictor import GesturePredictor

def verify_random_test_images(project_root: Path, predictor: GesturePredictor, num_samples: int = 10, seed: int = 42):
    """Samples random images from metadata/test.csv and verifies prediction accuracy."""
    test_csv = project_root / "metadata" / "test.csv"
    if not test_csv.exists():
        print(f"Warning: {test_csv} not found, skipping random verification.")
        return

    df = pd.read_csv(test_csv)
    sample_df = df.sample(n=num_samples, random_state=seed)

    print("\n" + "=" * 70)
    print(f" RANDOM TEST VERIFICATION ({num_samples} Samples)")
    print("=" * 70)
    print(f"{'#':<3} | {'Ground Truth':<15} | {'Predicted Label':<15} | {'Confidence':<12} | {'Result':<6}")
    print("-" * 70)

    correct = 0
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_p = project_root / row["image_path"]
        gt_label = row["label"]
        with Image.open(img_p) as img:
            gray = img.convert("L").resize((50, 50))
            arr = np.array(gray, dtype=np.float32)
            input_tensor = np.expand_dims(np.expand_dims(arr, axis=0), axis=-1)

        res = predictor.predict(input_tensor)
        pred_label = res["label"]
        conf = res["confidence"] * 100.0

        is_match = (gt_label == pred_label)
        if is_match:
            correct += 1
        res_str = "MATCH" if is_match else "DIFF"

        print(f"{idx+1:<3} | {gt_label:<15} | {pred_label:<15} | {conf:6.2f}%      | {res_str}")

    print("-" * 70)
    print(f"Batch Verification Accuracy: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Predict Sign Language Gesture from Image")
    parser.add_argument("--image", type=str, default="", help="Path to single image file")
    parser.add_argument("--model", type=str, default="", help="Path to trained model (.keras)")
    parser.add_argument("--top-k", type=int, default=1, help="Show top-k predictions")
    parser.add_argument("--verify-random", type=int, default=0, help="Number of random test samples to verify")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model) if args.model else (project_root / "models" / "diagnostic_modern_cnn.keras")
    
    # Fallback to diagnostic_adam or baseline if modern_cnn not present
    if not model_path.exists():
        fallback_candidates = [
            project_root / "models" / "diagnostic_adam.keras",
            project_root / "models" / "sign_language_cnn.keras"
        ]
        for fb in fallback_candidates:
            if fb.exists():
                model_path = fb
                break

    predictor = GesturePredictor(model_path=model_path)

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = project_root / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            gray = img.convert("L").resize((50, 50))
            arr = np.array(gray, dtype=np.float32)
            input_tensor = np.expand_dims(np.expand_dims(arr, axis=0), axis=-1)

        res = predictor.predict(input_tensor)
        print(f"Predicted label: {res['label']}")
        print(f"Class ID: {res['class_id']}")
        print(f"Confidence: {res['confidence']*100.0:.2f}%")

        if args.top_k > 1:
            top_k = predictor.predict_top_k(input_tensor, k=args.top_k)
            print(f"\nTop-{args.top_k} Predictions:")
            for item in top_k:
                print(f"  {item['rank']}. {item['label']:<15} (ID: {item['class_id']:2d}) - {item['confidence']*100.0:6.2f}%")

    if args.verify_random > 0 or not args.image:
        num_samples = args.verify_random if args.verify_random > 0 else 10
        verify_random_test_images(project_root, predictor, num_samples=num_samples)

if __name__ == "__main__":
    main()
