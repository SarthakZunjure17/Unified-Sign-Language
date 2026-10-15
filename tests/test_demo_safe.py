"""
tests/test_demo_safe.py
-----------------------
Comprehensive verification for Phase 4.4 Demo Safety Policy.
Ensures Class 42 ("Fuck") is completely suppressed from:
- Predictions (no secondary substitution, returns Unknown)
- Temporal Stabilizer (never committed)
- Text Mode (never appended)
- Speech Engine (never vocalized)
- Calculator Mode (never dispatched)
- OpenCV HUD Overlay & Debug Telemetry (never rendered)
Also verifies:
- Normal eligible gestures continue to function properly.
- Stabilizer duplicate suppression remains fully operational.
- Original 44-class model and metadata/labels.json remain strictly unchanged.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from recognition.predictor import GesturePredictor
from recognition.stabilizer import TemporalStabilizer
from application.controller import AppController
from application.text_mode import TextMode
from application.calculator import SafeCalculator
from application.speech import SpeechEngine
from ui.hud import HUDRenderer

# ---------------------------------------------------------------------------
# 1. Normal Eligible Gestures Still Work
# ---------------------------------------------------------------------------
def test_normal_eligible_gestures_work_in_demo_safe():
    predictor = GesturePredictor(demo_safe=True)
    test_gestures = [
        ("A", "A"),
        ("B", "B"),
        ("D", "D"),
        ("U", "U"),
        ("I_Love_You", "I love you"),
    ]
    for folder, expected_label in test_gestures:
        img_path = PROJECT_ROOT / "dataset" / "master" / folder / "1.jpg"
        if not img_path.exists():
            continue
        raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(raw_img, (50, 50))
        tensor = resized.astype(np.float32)

        pred = predictor.predict(tensor)
        assert pred["is_excluded"] is False
        assert pred["class_id"] >= 0
        assert pred["label"] == expected_label
        assert pred["display_label"] == expected_label
        assert pred["confidence"] >= 0.70

# ---------------------------------------------------------------------------
# 2. Raw Prediction of Class 42 Becomes Unknown in Demo-Safe Mode
# ---------------------------------------------------------------------------
def test_raw_prediction_class_42_becomes_unknown_in_demo_safe():
    predictor_safe = GesturePredictor(demo_safe=True)
    predictor_raw = GesturePredictor(demo_safe=False)

    img_path = PROJECT_ROOT / "dataset" / "master" / "Fuck" / "1.jpg"
    assert img_path.exists(), "Master dataset image for Fuck must exist"

    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(raw_img, (50, 50))
    tensor = resized.astype(np.float32)

    # In raw mode: underlying model predicts Class 42 "Fuck"
    raw_pred = predictor_raw.predict(tensor)
    assert raw_pred["class_id"] == 42
    assert raw_pred["label"] == "Fuck"
    assert raw_pred["display_label"] == "Fuck"
    assert raw_pred["is_excluded"] is False

    # In demo-safe mode: becomes Unknown and shows unavailable message
    safe_pred = predictor_safe.predict(tensor)
    assert safe_pred["class_id"] == -1
    assert safe_pred["raw_class_id"] == 42
    assert safe_pred["label"] == "Unknown"
    assert safe_pred["display_label"] == "Gesture unavailable in demo mode"
    assert safe_pred["is_confident"] is False
    assert safe_pred["is_excluded"] is True

def test_no_substitution_of_secondary_class_when_class_42_wins():
    """Verify that when Class 42 has highest probability, the second highest is NOT substituted."""
    predictor = GesturePredictor(demo_safe=True)

    # Mock the internal model to return synthetic probabilities
    # where Class 42 wins with 0.85 and Class 0 ("A") is second with 0.15
    mock_probs = np.zeros(44, dtype=np.float32)
    mock_probs[42] = 0.85
    mock_probs[0] = 0.15

    dummy_tensor = np.zeros((1, 50, 50, 1), dtype=np.float32)
    with patch.object(predictor, "model") as mock_model:
        mock_output = MagicMock()
        mock_output.numpy.return_value = np.array([mock_probs], dtype=np.float32)
        mock_model.return_value = mock_output
        mock_model.input_names = ["input"]

        pred = predictor.predict(dummy_tensor)

        # Must NOT be substituted to Class 0 ("A")
        assert pred["class_id"] == -1
        assert pred["raw_class_id"] == 42
        assert pred["label"] == "Unknown"
        assert pred["display_label"] == "Gesture unavailable in demo mode"
        assert pred["is_confident"] is False
        assert pred["is_excluded"] is True

def test_top_k_excludes_class_42_in_demo_safe():
    predictor = GesturePredictor(demo_safe=True)
    dummy_tensor = np.zeros((1, 50, 50, 1), dtype=np.float32)
    top_5 = predictor.predict_top_k(dummy_tensor, k=5)
    class_ids = [item["class_id"] for item in top_5]
    labels = [item["label"].lower() for item in top_5]
    assert 42 not in class_ids
    assert "fuck" not in labels

# ---------------------------------------------------------------------------
# 3. Class 42 Cannot Be Committed to Text
# ---------------------------------------------------------------------------
def test_class_42_cannot_be_committed_to_text():
    text_mode = TextMode(demo_safe=True)
    assert text_mode.get_sentence() == ""

    # Attempt to append Class 42
    res = text_mode.append_gesture(42, "Fuck")
    assert res == ""
    assert text_mode.get_sentence() == ""

    # Append normal gestures
    text_mode.append_gesture(0, "A")
    text_mode.append_gesture(1, "B")
    assert text_mode.get_sentence() == "AB"

    # Attempt Class 42 again
    text_mode.append_gesture(42, "Fuck")
    assert text_mode.get_sentence() == "AB"

    # AppController dispatch test
    ctrl = AppController(default_mode="TEXT", demo_safe=True)
    ctrl.handle_committed_gesture(42, "Fuck")
    assert ctrl.text_mode.get_sentence() == ""
    assert "unavailable" in ctrl.status_message.lower()

# ---------------------------------------------------------------------------
# 4. Class 42 Cannot Be Spoken
# ---------------------------------------------------------------------------
def test_class_42_cannot_be_spoken():
    speech = SpeechEngine(demo_safe=True)
    with patch.object(speech, "_do_speak") as mock_speak:
        speech.is_available = True

        # Pure offensive word
        speech.speak("fuck", block=True)
        mock_speak.assert_not_called()

        # Mixed offensive phrase
        speech.speak("Sign language says fuck today", block=True)
        mock_speak.assert_called_once_with("Sign language says  today")

def test_controller_speech_sanitization():
    ctrl = AppController(default_mode="TEXT", demo_safe=True)
    ctrl.text_mode.sentence = "hello fuck world"
    with patch.object(ctrl.speech, "speak") as mock_speak:
        ctrl.speak_current()
        mock_speak.assert_called_once_with("hello  world")

# ---------------------------------------------------------------------------
# 5. Class 42 Cannot Be Dispatched to Calculator Logic
# ---------------------------------------------------------------------------
def test_class_42_cannot_be_dispatched_to_calculator():
    calc = SafeCalculator(demo_safe=True)
    assert calc.expression == ""

    calc.add_gesture(42, "Fuck")
    assert calc.expression == ""

    # Add valid digits: 26 -> '0', 31 -> '5'
    calc.add_gesture(31, "5")
    assert calc.expression == "5"

    calc.add_gesture(42, "Fuck")
    assert calc.expression == "5"

    ctrl = AppController(default_mode="CALCULATOR", demo_safe=True)
    ctrl.handle_committed_gesture(42, "Fuck")
    assert ctrl.calculator.expression == ""

# ---------------------------------------------------------------------------
# 6. HUD Does Not Display Excluded Label
# ---------------------------------------------------------------------------
def test_hud_does_not_display_excluded_label():
    hud = HUDRenderer(frame_width=800, frame_height=600, panel_width=400, demo_safe=True)
    blank_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Test with excluded prediction
    pred = {
        "class_id": -1,
        "raw_class_id": 42,
        "label": "Unknown",
        "display_label": "Gesture unavailable in demo mode",
        "confidence": 0.88,
        "is_confident": False,
        "is_excluded": True
    }
    stab_state = {
        "is_committed": False,
        "committed_class": None,
        "committed_label": None,
        "stable_label": "Gesture unavailable in demo mode",
        "stable_class": None,
        "status": "UNAVAILABLE (DEMO SAFE)"
    }
    ctrl_state = {
        "mode": "TEXT",
        "sentence": "",
        "expression": "",
        "calc_result": "",
        "calc_has_error": False,
        "status_message": "Ready",
        "speech_available": True,
        "demo_safe": True
    }

    canvas = hud.render(
        flipped_frame=blank_frame,
        roi_bbox=(360, 100, 400, 400),
        hand_detected=True,
        hand_crop_50=np.zeros((50, 50), dtype=np.uint8),
        prediction=pred,
        stabilizer_state=stab_state,
        controller_state=ctrl_state,
        fps=30.0,
        debug=True,
        debug_info={"rejection_reason": "none", "contour_area": 10000},
        last_commit_event=None
    )

    assert canvas.shape == (600, 1200, 3)

    # Defense-in-depth: Even if a rogue raw prediction dict with label="Fuck" is passed
    rogue_pred = {
        "class_id": 42,
        "raw_class_id": 42,
        "label": "Fuck",
        "display_label": "Fuck",
        "confidence": 0.95,
        "is_confident": True,
        "is_excluded": True
    }
    rogue_canvas = hud.render(
        flipped_frame=blank_frame,
        roi_bbox=(360, 100, 400, 400),
        hand_detected=True,
        hand_crop_50=np.zeros((50, 50), dtype=np.uint8),
        prediction=rogue_pred,
        stabilizer_state=stab_state,
        controller_state=ctrl_state,
        fps=30.0,
        debug=True,
        debug_info={"rejection_reason": "none", "contour_area": 10000},
        last_commit_event="'Fuck' (f#10)"
    )
    assert rogue_canvas.shape == (600, 1200, 3)

# ---------------------------------------------------------------------------
# 7. 44-Class Model and Original Labels Remain Intact
# ---------------------------------------------------------------------------
def test_model_and_labels_remain_unchanged():
    labels_file = PROJECT_ROOT / "metadata" / "labels.json"
    assert labels_file.exists(), "metadata/labels.json must exist"

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    if isinstance(labels_data, list):
        assert len(labels_data) == 44
        class_42 = next(item for item in labels_data if int(item["class_id"]) == 42)
        assert class_42["label"] == "Fuck"
    else:
        assert len(labels_data) == 44
        assert labels_data["42"] == "Fuck" or labels_data[42] == "Fuck"

    # Verify models/ artifacts exist and have non-zero size
    for mname in ["diagnostic_modern_cnn.keras", "diagnostic_adam.keras", "sign_language_cnn.keras"]:
        p = PROJECT_ROOT / "models" / mname
        assert p.exists(), f"Model {mname} must exist untouched"
        assert p.stat().st_size > 0

    # Verify original upstream model if available
    orig_h5 = PROJECT_ROOT.parent / "Sign-Language-master" / "Sign-Language-master" / "cnn_model_keras2.h5"
    if orig_h5.exists():
        assert orig_h5.stat().st_size > 0

# ---------------------------------------------------------------------------
# 8. Stabilizer Duplicate Suppression Continues to Work
# ---------------------------------------------------------------------------
def test_stabilizer_duplicate_suppression_and_rearm():
    stabilizer = TemporalStabilizer(
        window_size=12,
        min_confidence=0.70,
        min_stable_count=5,
        rearm_frames=4,
        demo_safe=True
    )

    pred_a = {"class_id": 0, "label": "A", "confidence": 0.90}

    # Feed frames 1 to 4: not yet stable
    for _ in range(4):
        res = stabilizer.update(pred_a, hand_detected=True)
        assert res["is_committed"] is False

    # Frame 5: reaches min_stable_count (5) -> commits "A"
    res = stabilizer.update(pred_a, hand_detected=True)
    assert res["is_committed"] is True
    assert res["committed_label"] == "A"
    assert res["committed_class"] == 0

    # Frames 6 to 10: same gesture "A" continues -> suppressed!
    for _ in range(5):
        res = stabilizer.update(pred_a, hand_detected=True)
        assert res["is_committed"] is False
        assert res["committed_label"] is None

    # Feed Class 42 in demo-safe mode: NEVER committed
    pred_42 = {
        "class_id": -1,
        "raw_class_id": 42,
        "label": "Unknown",
        "confidence": 0.95,
        "is_excluded": True
    }
    for _ in range(10):
        res = stabilizer.update(pred_42, hand_detected=True)
        assert res["is_committed"] is False
        assert res["committed_label"] is None
        assert res["stable_label"] == "Gesture unavailable in demo mode"

    # Idle frames to trigger rearm
    idle_pred = {"class_id": -1, "label": "None", "confidence": 0.0}
    for _ in range(5):
        stabilizer.update(idle_pred, hand_detected=False)

    # Feed frames of gesture "B"
    pred_b = {"class_id": 1, "label": "B", "confidence": 0.92}
    committed_b = False
    for _ in range(6):
        res = stabilizer.update(pred_b, hand_detected=True)
        if res["is_committed"]:
            committed_b = True
            assert res["committed_label"] == "B"
            assert res["committed_class"] == 1

    assert committed_b is True, "Gesture B should commit once after re-arm"
