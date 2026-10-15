#!/usr/bin/env python3
"""
scripts/test_webcam_pipeline.py
-------------------------------
End-to-end automated verification of the real-time webcam and inference pipeline.
Tests without requiring a physical camera, and probes physical camera availability.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.preprocessing import HandPreprocessor
from recognition.predictor import GesturePredictor
from recognition.stabilizer import TemporalStabilizer
from application.controller import AppController
from ui.hud import HUDRenderer

def test_pipeline():
    print("=" * 70)
    print(" AUTOMATED WEBCAM PIPELINE & INFERENCE VERIFICATION")
    print("=" * 70)

    # 1. Model & Label Loading
    print("1. Testing Model and Label Loading...")
    predictor = GesturePredictor()
    assert len(predictor.labels) == 44, f"Expected 44 labels, got {len(predictor.labels)}"
    assert predictor.labels[42] == "Fuck", f"Class 42 is '{predictor.labels[42]}', expected 'Fuck'"
    print("   [PASS] 44 classes verified. Class 42 is 'Fuck'. Model loaded.")

    # 2. Preprocessor Initialization
    print("2. Testing HandPreprocessor Initialization...")
    preprocessor = HandPreprocessor()
    print("   [PASS] Preprocessor initialized.")

    # 3. Test on Blank / Synthetic Frame (No Hand)
    print("3. Testing Processing on Blank Image (No Hand)...")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    prep_res = preprocessor.process_frame(blank_frame)
    assert not prep_res["hand_detected"], "Hand should not be detected on blank image"
    assert prep_res["processed_image"].shape == (1, 50, 50, 1)
    
    # Prediction on blank image should not crash
    pred_res = predictor.predict(prep_res["processed_image"])
    assert 0 <= pred_res["class_id"] <= 43 or pred_res["class_id"] == -1
    print("   [PASS] Blank image processed gracefully without crash.")

    # 4. Test on Actual Sample Image from Dataset
    print("4. Testing Preprocessing & Inference on Sample Dataset Image...")
    sample_path = PROJECT_ROOT / "dataset" / "master" / "Fuck" / "1.jpg"
    if not sample_path.exists():
        # Fallback to any file in dataset
        candidates = list((PROJECT_ROOT / "dataset" / "master").glob("*/*.jpg"))
        assert len(candidates) > 0, "No sample images found in dataset/master"
        sample_path = candidates[0]

    sample_bgr = cv2.imread(str(sample_path))
    assert sample_bgr is not None, f"Failed to load sample image: {sample_path}"
    
    # Place sample into synthetic webcam frame (800x600)
    synth_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    h, w = sample_bgr.shape[:2]
    # Place inside ROI region (360, 100)
    synth_frame[100:100+h, 360:360+w] = sample_bgr

    prep_synth = preprocessor.process_frame(synth_frame)
    pred_synth = predictor.predict(prep_synth["processed_image"])

    assert 0 <= pred_synth["class_id"] <= 43, f"Invalid class ID: {pred_synth['class_id']}"
    assert pred_synth["label"] in predictor.labels.values(), f"Invalid label: {pred_synth['label']}"
    assert 0.0 <= pred_synth["confidence"] <= 1.0, f"Confidence out of range: {pred_synth['confidence']}"
    print(f"   [PASS] Sample Image Prediction: Class {pred_synth['class_id']} ({pred_synth['label']}) | Confidence: {pred_synth['confidence']*100:.2f}%")

    # 5. Temporal Stabilizer & Controller
    print("5. Testing Temporal Stabilizer & AppController...")
    stabilizer = TemporalStabilizer(window_size=12, min_stable_count=7)
    controller = AppController(default_mode="TEXT")

    # Simulate 10 frames of 'A' (Class 0)
    sim_pred_A = {"class_id": 0, "label": "A", "confidence": 0.99, "is_confident": True}
    commit_count = 0
    for _ in range(10):
        s_res = stabilizer.update(sim_pred_A, hand_detected=True)
        if s_res["is_committed"]:
            commit_count += 1
            controller.handle_committed_gesture(s_res["committed_class"], s_res["committed_label"])

    assert commit_count == 1, f"Duplicate suppression failed: committed {commit_count} times, expected 1"
    assert controller.text_mode.get_sentence() == "A", f"Sentence mismatch: '{controller.text_mode.get_sentence()}'"
    print("   [PASS] Temporal stabilization and duplicate suppression verified.")

    # 6. Safe Calculator
    print("6. Testing Calculator Mode...")
    controller.set_mode = "CALCULATOR"
    controller.mode = "CALCULATOR"
    controller.calculator.clear()
    controller.calculator.add_token("25+75")
    _, res = controller.calculator.calculate()
    assert res == "100", f"Calculator evaluation error: expected '100', got '{res}'"
    print("   [PASS] Safe calculator evaluated '25+75 = 100'.")

    # 7. HUD Renderer
    print("7. Testing HUD Rendering Engine...")
    hud = HUDRenderer()
    hud_canvas = hud.render(
        flipped_frame=synth_frame,
        roi_bbox=preprocessor.roi_bbox,
        hand_detected=True,
        hand_crop_50=np.zeros((50, 50), dtype=np.uint8),
        prediction=pred_synth,
        stabilizer_state={"status": "STABLE"},
        controller_state=controller.get_display_state(),
        fps=30.0
    )
    assert hud_canvas.shape == (600, 1200, 3), f"HUD dimensions mismatch: {hud_canvas.shape}"
    print("   [PASS] HUD canvas generated successfully (1200x600).")

    # 8. Physical Camera Hardware Probe
    print("\n8. Probing Physical Webcam Hardware...")
    cam_test = cv2.VideoCapture(0)
    has_cam = cam_test.isOpened()
    if has_cam:
        ret, f = cam_test.read()
        cam_test.release()
        if ret and f is not None:
            print("   [INFO] Physical webcam detected and frame grab succeeded at camera index 0.")
        else:
            print("   [INFO] Camera device opened but frame grab timed out/empty.")
    else:
        print("   [INFO] No physical webcam device detected or camera busy.")

    print("\n" + "=" * 70)
    print(" PIPELINE VALIDATION RESULT:")
    print(" Automated pipeline passed; physical webcam validation pending.")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
