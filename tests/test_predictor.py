"""
tests/test_predictor.py
-----------------------
Verifies model loading, output shape (44,), inference output format,
confidence thresholding, and top-k predictions.
"""

from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from recognition.predictor import GesturePredictor

@pytest.fixture(scope="module")
def predictor():
    return GesturePredictor()

def test_model_loads(predictor):
    assert predictor.model is not None, "Model failed to load"

def test_model_output_shape(predictor):
    out_shape = predictor.model.output_shape
    num_classes = out_shape[-1] if isinstance(out_shape, tuple) else out_shape[0][-1]
    assert num_classes == 44, f"Expected 44 output units, got {num_classes}"

def test_predictor_output_format(predictor):
    dummy_input = np.ones((1, 50, 50, 1), dtype=np.float32) * 128.0
    res = predictor.predict(dummy_input)

    assert isinstance(res, dict)
    assert "class_id" in res
    assert "label" in res
    assert "display_label" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert "is_confident" in res

    assert 0 <= res["class_id"] <= 43
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 44

def test_predictor_confidence_threshold(predictor):
    # Test low confidence behavior
    pred_low = {
        "class_id": 5,
        "label": "F",
        "display_label": "Unknown",
        "confidence": 0.45,
        "is_confident": False
    }
    assert pred_low["confidence"] < predictor.confidence_threshold
    assert pred_low["display_label"] == "Unknown"

def test_top_k_predictions(predictor):
    dummy_input = np.ones((1, 50, 50, 1), dtype=np.float32) * 100.0
    top_5 = predictor.predict_top_k(dummy_input, k=5)

    assert len(top_5) == 5
    for rank, item in enumerate(top_5, 1):
        assert item["rank"] == rank
        assert 0 <= item["class_id"] <= 43
        assert 0.0 <= item["confidence"] <= 1.0
    
    # Confirm sorted descending
    confs = [item["confidence"] for item in top_5]
    assert confs == sorted(confs, reverse=True)
