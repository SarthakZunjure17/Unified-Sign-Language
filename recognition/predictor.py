"""
recognition/predictor.py
------------------------
Model inference wrapper for Unified Sign Language Interpreter.
Loads models/diagnostic_modern_cnn.keras once at startup and reads labels
exclusively from metadata/labels.json.
Supports configurable application-level Demo Safety Policy to exclude offensive
gestures (Class 42) during presentations without modifying underlying model weights.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import numpy as np
import tensorflow as tf

# Canonical excluded classes for classroom demo mode
DEFAULT_EXCLUDED_CLASSES = {42}

class GesturePredictor:
    """
    Handles model inference, label mapping, confidence thresholding,
    and configurable demo safety policy.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        confidence_threshold: float = 0.70,
        demo_safe: bool = True,
        excluded_classes: Optional[Set[int]] = None
    ):
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = model_path or (project_root / "models" / "diagnostic_modern_cnn.keras")
        self.labels_path = labels_path or (project_root / "metadata" / "labels.json")
        self.confidence_threshold = confidence_threshold
        self.demo_safe = demo_safe
        self.excluded_classes = excluded_classes if excluded_classes is not None else DEFAULT_EXCLUDED_CLASSES

        self.labels = self._load_and_verify_labels()
        self.model = self._load_model()

    def _load_and_verify_labels(self) -> Dict[int, str]:
        """Loads and strictly verifies 44 classes with Class 42 == 'Fuck'."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        id_to_label = {}
        if isinstance(data, list):
            for item in data:
                cid = int(item["class_id"])
                id_to_label[cid] = item["label"]
        elif isinstance(data, dict):
            for k, v in data.items():
                id_to_label[int(k)] = v
        else:
            raise ValueError(f"Unexpected JSON format in {self.labels_path}")

        # Strict checks required by specifications
        if len(id_to_label) != 44:
            raise ValueError(f"Label verification failed: expected 44 classes, got {len(id_to_label)}")
        
        if id_to_label.get(42) != "Fuck":
            raise ValueError(f"Label verification failed: class 42 is '{id_to_label.get(42)}', MUST be 'Fuck'")

        return id_to_label

    def _load_model(self) -> tf.keras.Model:
        """Loads the trained Keras model once."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}")

        print(f"Loading gesture recognition model: {self.model_path.name}...")
        model = tf.keras.models.load_model(str(self.model_path))
        
        # Verify output dimension
        out_shape = model.output_shape
        num_classes = out_shape[-1] if isinstance(out_shape, tuple) else out_shape[0][-1]
        if num_classes != 44:
            raise ValueError(f"Model output dimension mismatch: expected 44, got {num_classes}")

        # Warm up model to eliminate first-frame latency
        dummy = np.zeros((1, 50, 50, 1), dtype=np.float32)
        _ = model(dummy, training=False)
        print("Model loaded and warmed up successfully.")
        return model

    def predict(self, image_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single 50x50x1 image array.
        Input pixel range must be [0, 255] (float32 or uint8).
        In demo-safe mode, if class 42 is top prediction, returns Unknown.
        Does NOT substitute secondary classes.
        """
        if image_tensor is None or image_tensor.size == 0:
            return {
                "class_id": -1,
                "raw_class_id": -1,
                "label": "None",
                "display_label": "No Hand",
                "confidence": 0.0,
                "probabilities": np.zeros(44, dtype=np.float32),
                "is_confident": False,
                "is_excluded": False
            }

        # Ensure shape (1, 50, 50, 1)
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, axis=0)
        elif len(image_tensor.shape) == 2:
            image_tensor = np.expand_dims(np.expand_dims(image_tensor, axis=0), axis=-1)

        # Fast direct forward pass
        inp_name = self.model.input_names[0] if hasattr(self.model, "input_names") and self.model.input_names else None
        if inp_name:
            probs = self.model({inp_name: image_tensor}, training=False).numpy()[0]
        else:
            probs = self.model(image_tensor, training=False).numpy()[0]
        pred_class = int(np.argmax(probs))
        raw_label = self.labels.get(pred_class, f"Class_{pred_class}")
        confidence = float(probs[pred_class])
        is_confident = confidence >= self.confidence_threshold

        # Demo-Safe Policy: Exclude specified classes (e.g. Class 42)
        if self.demo_safe and pred_class in self.excluded_classes:
            return {
                "class_id": -1,
                "raw_class_id": pred_class,
                "label": "Unknown",
                "display_label": "Gesture unavailable in demo mode",
                "confidence": confidence,
                "probabilities": probs,
                "is_confident": False,
                "is_excluded": True
            }

        return {
            "class_id": pred_class,
            "raw_class_id": pred_class,
            "label": raw_label,
            "display_label": raw_label if is_confident else "Unknown",
            "confidence": confidence,
            "probabilities": probs,
            "is_confident": is_confident,
            "is_excluded": False
        }

    def predict_top_k(self, image_tensor: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Returns top-k predictions sorted by confidence descending, filtering excluded classes in demo mode."""
        pred_res = self.predict(image_tensor)
        probs = pred_res["probabilities"]
        top_indices = np.argsort(probs)[::-1]

        top_k = []
        for idx in top_indices:
            cid = int(idx)
            # In demo-safe mode, exclude offensive classes from top-k inspection lists
            if self.demo_safe and cid in self.excluded_classes:
                continue
            top_k.append({
                "rank": len(top_k) + 1,
                "class_id": cid,
                "label": self.labels.get(cid, f"Class_{cid}"),
                "confidence": float(probs[cid])
            })
            if len(top_k) >= k:
                break
        return top_k
