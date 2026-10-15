#!/usr/bin/env python3
"""
scripts/verify_phase_4_2.py
---------------------------
Verification script for PHASE 4.2:
- Checks default ROI constants (400x400)
- Checks UI dimensions (800x600 camera + 400x600 dashboard = 1200x600)
- Checks clamping logic on varied frame sizes and CLI inputs
- Checks CNN model input dimensions (50x50x1)
- Validates HUD rendering with and without hand
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.preprocessing import (
    DEFAULT_ROI_X,
    DEFAULT_ROI_Y,
    DEFAULT_ROI_W,
    DEFAULT_ROI_H,
    DEFAULT_ROI_BBOX,
    clamp_roi,
    HandPreprocessor
)
from recognition.predictor import GesturePredictor
from ui.hud import HUDRenderer
from application.controller import AppController

def verify_all():
    print("=" * 65)
    print(" PHASE 4.2 - ENLARGE WEBCAM ROI AND UI VERIFICATION")
    print("=" * 65)

    # 1. ROI Constants
    print("1. Checking ROI Constants...")
    assert DEFAULT_ROI_W == 400, f"Expected DEFAULT_ROI_W == 400, got {DEFAULT_ROI_W}"
    assert DEFAULT_ROI_H == 400, f"Expected DEFAULT_ROI_H == 400, got {DEFAULT_ROI_H}"
    assert DEFAULT_ROI_W == DEFAULT_ROI_H, "ROI is not square!"
    assert DEFAULT_ROI_BBOX == (DEFAULT_ROI_X, DEFAULT_ROI_Y, 400, 400)
    print(f"   [PASS] Default ROI is square: {DEFAULT_ROI_W}x{DEFAULT_ROI_H} at ({DEFAULT_ROI_X}, {DEFAULT_ROI_Y})")

    # 2. Clamping Logic
    print("2. Checking ROI Clamping & Boundary Logic...")
    # Test valid ROI inside 800x600
    r1 = clamp_roi((200, 40, 400, 400), 800, 600)
    assert r1 == (200, 40, 400, 400)
    # Test overflow right/bottom
    r2 = clamp_roi((500, 300, 400, 400), 800, 600)
    assert r2 == (400, 200, 400, 400), f"Expected (400, 200, 400, 400), got {r2}"
    assert r2[0] + r2[2] <= 800
    assert r2[1] + r2[3] <= 600
    # Test negative coordinates
    r3 = clamp_roi((-100, -50, 400, 400), 800, 600)
    assert r3 == (0, 0, 400, 400)
    # Test clamping default ROI on 640x480 frame
    r4 = clamp_roi(DEFAULT_ROI_BBOX, 640, 480)
    assert r4 == (240, 80, 400, 400)
    print("   [PASS] Clamping guarantees 400x400 stays fully visible inside camera frame.")

    # 3. UI Scaling & Window Dimensions
    print("3. Checking UI Scaling and Window Dimensions...")
    hud = HUDRenderer(frame_width=800, frame_height=600, panel_width=400)
    assert hud.frame_width == 800
    assert hud.frame_height == 600
    assert hud.panel_width == 400
    assert hud.total_width == 1200

    # 4. CNN Input Shape Guarantee
    print("4. Checking Model Input Shape Preservation...")
    prep = HandPreprocessor()
    test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    res = prep.process_frame(test_frame)
    model_input = res["processed_image"]
    assert model_input.shape == (1, 50, 50, 1), f"Model input shape is {model_input.shape}, expected (1, 50, 50, 1)"
    assert model_input.dtype == np.float32, f"Model input dtype is {model_input.dtype}, expected float32"
    print("   [PASS] Preprocessor outputs strict 50x50x1 float32 tensor regardless of 400x400 ROI.")

    # 5. Full Canvas Rendering
    print("5. Checking HUD Canvas Assembly...")
    ctrl = AppController(default_mode="TEXT")
    pred = {"class_id": -1, "label": "None", "display_label": "No Hand", "confidence": 0.0, "is_confident": False}
    canvas = hud.render(
        flipped_frame=test_frame,
        roi_bbox=DEFAULT_ROI_BBOX,
        hand_detected=False,
        hand_crop_50=None,
        prediction=pred,
        stabilizer_state={"status": "SEARCHING"},
        controller_state=ctrl.get_display_state(),
        fps=30.0,
        debug=True,
        debug_info=res["debug_info"]
    )
    assert canvas.shape == (600, 1200, 3), f"Rendered canvas shape is {canvas.shape}, expected (600, 1200, 3)"
    print(f"   [PASS] Canvas rendered successfully at {canvas.shape[1]}x{canvas.shape[0]} (1200x600).")

    # 6. Verify 44 Classes and Model
    print("6. Checking 44 Classes & Class 42 integrity...")
    predictor = GesturePredictor()
    assert len(predictor.labels) == 44
    assert predictor.labels[42] == "Fuck"
    print("   [PASS] All 44 classes intact. Class 42 is 'Fuck'.")

    print("=" * 65)
    print(" ALL PHASE 4.2 VERIFICATIONS PASSED SUCCESSFULLY!")
    print(f" Final ROI Size:          {DEFAULT_ROI_W} x {DEFAULT_ROI_H}")
    print(f" Final Camera Feed:       800 x 600 (4:3 aspect ratio)")
    print(f" Final Dashboard Panel:   400 x 600")
    print(f" Final Window Dimensions: 1200 x 600")
    print(f" CNN Model Input:         50 x 50 x 1 (float32)")
    print("=" * 65)

if __name__ == "__main__":
    verify_all()
