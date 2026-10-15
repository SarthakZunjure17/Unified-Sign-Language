"""
tests/test_preprocessing.py
--------------------------
Verifies preprocessing pipeline rejection rules and validation:
- no contour -> No Hand
- tiny contour -> No Hand
- extremely thin contour -> No Hand
- valid hand-like contour -> Hand Detected
"""

from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from recognition.preprocessing import HandPreprocessor, DEFAULT_ROI_BBOX

@pytest.fixture
def preprocessor():
    return HandPreprocessor()

def test_no_contour_returns_no_hand(preprocessor):
    """Verifies that an empty/blank frame results in No Hand."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = preprocessor.process_frame(blank)
    assert not res["hand_detected"]
    assert res["contour_bbox"] is None
    assert np.all(res["processed_image"] == 0)
    assert res["debug_info"]["rejection_reason"] in ["Empty frame", "No contour found"]

def test_tiny_contour_returns_no_hand(preprocessor):
    """Verifies that tiny background noise specks are rejected as No Hand."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Circle with radius 3 (area ~ 28 px), well below MIN_CONTOUR_AREA (3500 px)
    cv2.circle(frame, (200, 200), 3, (255, 255, 255), -1)
    res = preprocessor.process_frame(frame)
    assert not res["hand_detected"]
    assert np.all(res["processed_image"] == 0)
    assert "small" in res["debug_info"]["rejection_reason"].lower()

def test_extremely_thin_contour_returns_no_hand(preprocessor):
    """Verifies that thin/elongated contours (vertical lines, horizontal edges) are rejected."""
    # 1. Very thin vertical strip (e.g. door seam / cable / shadow)
    frame_v = np.zeros((480, 640, 3), dtype=np.uint8)
    # Rectangle of width 18, height 240
    cv2.rectangle(frame_v, (200, 100), (218, 340), (255, 255, 255), -1)
    res_v = preprocessor.process_frame(frame_v)
    assert not res_v["hand_detected"]
    assert np.all(res_v["processed_image"] == 0)

    # 2. Narrow elongated contour with aspect ratio below 0.28
    frame_narrow = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_narrow, (150, 100), (196, 320), (255, 255, 255), -1)
    res_narrow = preprocessor.process_frame(frame_narrow)
    assert not res_narrow["hand_detected"]
    assert np.all(res_narrow["processed_image"] == 0)

    # 3. Very thin horizontal strip (e.g. desk / shelf line)
    frame_h = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame_h, (50, 200), (300, 218), (255, 255, 255), -1)
    res_h = preprocessor.process_frame(frame_h)
    assert not res_h["hand_detected"]
    assert np.all(res_h["processed_image"] == 0)

def test_valid_hand_like_contour_detected(preprocessor):
    """Verifies that a valid hand-sized and proportioned contour is accepted as Hand Detected."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Synthetic hand-like rectangle placed so that after horizontal flip it lands in ROI
    # Flipped frame ROI: x=300..600, y=80..380
    # Original coordinates: x=100..260 (maps to 379..539 flipped), y=140..300
    cv2.rectangle(frame, (100, 140), (260, 300), (255, 255, 255), -1)

    res = preprocessor.process_frame(frame)
    assert res["hand_detected"] is True
    assert res["contour_bbox"] is not None
    assert res["processed_image"].shape == (1, 50, 50, 1)
    assert res["processed_image"].dtype == np.float32
    assert np.max(res["processed_image"]) > 0
    assert res["debug_info"]["rejection_reason"] == "Valid Hand"

def test_preprocess_image_array():
    """Verifies standalone image array preprocessing helper."""
    img_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 150
    out = HandPreprocessor.preprocess_image_array(img_bgr)
    assert out.shape == (1, 50, 50, 1)
    assert out.dtype == np.float32

    img_gray = np.ones((80, 80), dtype=np.uint8) * 200
    out_gray = HandPreprocessor.preprocess_image_array(img_gray)
    assert out_gray.shape == (1, 50, 50, 1)

def test_clamp_roi_preserves_square_and_fits_frame():
    """Verifies that clamp_roi enforces frame boundaries and preserves square geometry."""
    from recognition.preprocessing import clamp_roi

    # 1. Standard valid 400x400 in 800x600 frame
    roi = clamp_roi((200, 40, 400, 400), 800, 600)
    assert roi == (200, 40, 400, 400)

    # 2. Negative coordinates clamped to 0
    roi_neg = clamp_roi((-50, -20, 400, 400), 800, 600)
    assert roi_neg == (0, 0, 400, 400)

    # 3. Exceeds right and bottom boundaries -> repositioned so full box is visible
    roi_overflow = clamp_roi((500, 300, 400, 400), 800, 600)
    # x max is 800 - 400 = 400, y max is 600 - 400 = 200
    assert roi_overflow == (400, 200, 400, 400)

    # 4. Oversized square clamped to min dimension (600 in 800x600)
    roi_huge = clamp_roi((0, 0, 900, 900), 800, 600)
    assert roi_huge == (0, 0, 600, 600)

    # 5. Clamping in smaller 640x480 frame
    roi_small = clamp_roi((360, 100, 400, 400), 640, 480)
    assert roi_small[2] == 400 and roi_small[3] == 400
    assert roi_small[0] + roi_small[2] <= 640
    assert roi_small[1] + roi_small[3] <= 480

def test_enlarged_400x400_roi_on_800x600_frame():
    """Verifies hand detection inside the enlarged 400x400 ROI on an 800x600 frame."""
    prep = HandPreprocessor(roi_bbox=(360, 100, 400, 400))
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Flipped frame ROI: x=360..760, y=100..500
    # Original coordinates before horizontal flip:
    # x in [100, 260] flips to x in [539, 699], which falls cleanly inside [360, 760]
    cv2.rectangle(frame, (100, 160), (260, 360), (255, 255, 255), -1)

    res = prep.process_frame(frame)
    assert res["hand_detected"] is True
    assert res["contour_bbox"] is not None
    # Crucial: Model still receives 50x50x1 float32
    assert res["processed_image"].shape == (1, 50, 50, 1)
    assert res["processed_image"].dtype == np.float32
    assert np.max(res["processed_image"]) > 0
    assert res["debug_info"]["rejection_reason"] == "Valid Hand"

def test_empty_background_rejection(preprocessor):
    """Verifies that an empty wall/table background with non-skin color is rejected as No Hand."""
    # Simulates a realistic neutral indoor wall (BGR [145, 150, 155])
    wall_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    wall_frame[:, :] = [145, 150, 155]
    res = preprocessor.process_frame(wall_frame)
    assert not res["hand_detected"]
    assert res["contour_bbox"] is None
    assert np.all(res["processed_image"] == 0)
    assert "no hand" in res["debug_info"]["rejection_reason"].lower()

def test_solid_silhouette_internal_holes_filled(preprocessor):
    """Verifies that internal dark holes/gaps inside a hand contour are filled solid white."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Outer white box (hand silhouette)
    cv2.rectangle(frame, (100, 160), (260, 360), (255, 255, 255), -1)
    # Inner black gap/hole (shadow or lighting crack)
    cv2.rectangle(frame, (140, 200), (200, 280), (0, 0, 0), -1)

    res = preprocessor.process_frame(frame)
    assert res["hand_detected"] is True
    # In the cropped 50x50 hand, the center should be solid white (filled), not a black hole
    crop_50 = res["hand_crop_50"]
    center_val = crop_50[25, 25]
    assert center_val == 255, f"Expected filled solid silhouette, got center value {center_val}"

def test_interactive_palm_calibration(preprocessor):
    """Verifies that palm calibration generates and stores a valid 2D HSV histogram."""
    # Synthetic palm patch with realistic skin color
    palm_roi = np.zeros((400, 400, 3), dtype=np.uint8)
    # BGR [125, 150, 205]
    palm_roi[150:250, 150:250] = [125, 150, 205]
    calib_res = preprocessor.calibrate_from_roi(palm_roi)
    assert calib_res["success"] is True
    assert preprocessor.hist is not None
    assert preprocessor.hist.shape == (180, 256)
    assert preprocessor.hist.max() == 255.0

def test_required_gestures_pipeline_inference(preprocessor):
    """Verifies required gestures (A, B, D, U, Fuck, I love you) pass preprocessing and inference under demo-safe & raw modes."""
    from recognition.predictor import GesturePredictor
    predictor_safe = GesturePredictor(demo_safe=True)
    predictor_raw = GesturePredictor(demo_safe=False)

    test_classes = ["A", "B", "D", "U", "Fuck", "I_Love_You"]
    for cls in test_classes:
        sample_path = PROJECT_ROOT / "dataset" / "master" / cls / "1.jpg"
        if not sample_path.exists():
            continue
        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        hand_scaled = cv2.resize(img, (220, 220), interpolation=cv2.INTER_NEAREST)
        hand_bgr = cv2.cvtColor(hand_scaled, cv2.COLOR_GRAY2BGR)
        frame[180:400, 100:320] = hand_bgr

        res = preprocessor.process_frame(frame)
        assert res["hand_detected"] is True, f"Gesture {cls} was not detected"

        expected_label = cls if cls != "I_Love_You" else "I love you"

        # Raw inference mode: original model classification remains intact
        raw_pred = predictor_raw.predict(res["processed_image"])
        assert raw_pred["label"] == expected_label, f"Raw expected {expected_label}, got {raw_pred['label']}"
        assert raw_pred["confidence"] >= 0.70

        # Demo-Safe mode: Class 42 returns Unknown, while safe gestures are recognized
        safe_pred = predictor_safe.predict(res["processed_image"])
        if cls == "Fuck":
            assert safe_pred["label"] == "Unknown"
            assert safe_pred["display_label"] == "Gesture unavailable in demo mode"
            assert safe_pred["class_id"] == -1
            assert safe_pred["is_excluded"] is True
        else:
            assert safe_pred["label"] == expected_label
            assert safe_pred["confidence"] >= 0.70


