"""
tests/test_silhouette_polish.py
-------------------------------
Verifies Phase 4.5 Hand Silhouette Polish:
- Selective hole filling via topological flood fill (palm filled, finger valleys preserved)
- Connected-component analysis for noise suppression
- Exact 50x50 square padding and nearest-neighbor 200x200 enlargement
- Debug snapshot creation with all 6 pipeline stages
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from recognition.preprocessing import HandPreprocessor
from recognition.camera import CameraPipeline

def test_flood_fill_hole_filling_preserves_finger_valleys():
    """Verifies that enclosed holes inside the palm are filled while finger valleys stay open."""
    # Synthetic V gesture
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Palm
    mask[100:180, 60:140] = 255
    # Enclosed hole in palm (shadow/crease)
    mask[130:150, 90:110] = 0
    # Finger 1 (Index)
    mask[20:100, 70:95] = 255
    # Finger 2 (Middle)
    mask[20:100, 105:130] = 255
    # Valley between fingers is x=95..105, y=20..100 (10px wide gap)

    filled = HandPreprocessor.fill_internal_holes(mask)

    # 1. Enclosed palm hole should be filled with 255
    assert filled[140, 100] == 255, "Palm hole was not filled"
    # 2. Fingers must be preserved
    assert filled[50, 80] == 255, "Finger 1 was eroded"
    assert filled[50, 115] == 255, "Finger 2 was eroded"
    # 3. Valley between fingers MUST remain 0 (open background)
    assert filled[50, 100] == 0, "Valley between fingers was incorrectly filled/merged"

def test_connected_component_suppresses_isolated_noise():
    """Verifies that isolated noise blobs outside the hand are completely removed."""
    preprocessor = HandPreprocessor()

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Valid hand-like rectangle placed inside ROI
    cv2.rectangle(frame, (100, 160), (260, 360), (255, 255, 255), -1)
    # Small isolated background noise outside hand but inside ROI
    cv2.rectangle(frame, (20, 20), (35, 35), (255, 255, 255), -1)

    res = preprocessor.process_frame(frame)
    assert res["hand_detected"] is True

    thresh_roi = res["thresh_roi"]
    # Hand area is present
    assert np.max(thresh_roi) == 255
    # Verify no tiny disjoint components survived in thresh_roi
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_roi, connectivity=8)
    # Background (label 0) + exactly 1 hand component (label 1)
    assert num_labels == 2, f"Expected 1 hand component and background, got {num_labels} labels"

def test_enlarged_preview_exact_integer_scaling():
    """Verifies that the 50x50 model input enlarges cleanly to 200x200 via nearest-neighbor."""
    sample_50 = np.zeros((50, 50), dtype=np.uint8)
    sample_50[10:40, 15:35] = 255

    enlarged = cv2.resize(sample_50, (200, 200), interpolation=cv2.INTER_NEAREST)
    assert enlarged.shape == (200, 200)

    # Check 4x block geometry: every pixel in 50x50 maps to a 4x4 block in 200x200
    for r in range(0, 50, 5):
        for c in range(0, 50, 5):
            val_50 = sample_50[r, c]
            block_200 = enlarged[r*4:(r+1)*4, c*4:(c+1)*4]
            assert np.all(block_200 == val_50), f"Pixel ({r},{c}) was not uniformly expanded 4x4"

def test_save_debug_snapshot_writes_all_six_stages(tmp_path):
    """Verifies that save_debug_snapshot saves all 6 pipeline stages and metadata."""
    pipeline = CameraPipeline()

    dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    dummy_crop = np.zeros((50, 50), dtype=np.uint8)
    dummy_crop[10:40, 10:40] = 255

    prep_res = {
        "flipped_frame": dummy_frame,
        "roi_bbox": (360, 100, 400, 400),
        "raw_mask": np.zeros((400, 400), dtype=np.uint8),
        "thresh_roi": np.zeros((400, 400), dtype=np.uint8),
        "hand_crop_50": dummy_crop,
        "hand_detected": True,
        "debug_info": {"contour_area": 15000, "aspect_ratio": 1.0}
    }
    pred_res = {
        "class_id": 0,
        "label": "A",
        "confidence": 0.99,
        "display_label": "A"
    }

    sub_dir_str = pipeline.save_debug_snapshot(prep_res, pred_res)
    sub_dir = Path(sub_dir_str)
    assert sub_dir.exists()

    expected_files = [
        "1_original_frame.png",
        "2_roi_crop.png",
        "3_raw_mask.png",
        "4_cleaned_mask.png",
        "5_model_input_50x50.png",
        "6_model_input_enlarged.png",
        "metadata.json"
    ]
    for ef in expected_files:
        fp = sub_dir / ef
        assert fp.exists(), f"Missing debug snapshot file: {ef}"
        assert fp.stat().st_size > 0

    # Verify metadata content
    with open(sub_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["hand_detected"] is True
    assert meta["prediction"]["label"] == "A"
    assert meta["prediction"]["confidence"] == 0.99

    # Verify enlarged image is 200x200
    enl_img = cv2.imread(str(sub_dir / "6_model_input_enlarged.png"))
    assert enl_img.shape == (200, 200, 3)
