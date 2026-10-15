#!/usr/bin/env python3
"""
scripts/debug_preprocessing.py
------------------------------
Visual and quantitative diagnostic tool for hand segmentation and CNN input preprocessing.
Compares live or saved webcam frames side-by-side with intermediate pipeline stages
and actual dataset training images.

Usage:
    # Inspect a saved image file without opening camera:
    python scripts/debug_preprocessing.py --image debug_out/raw_webcam.png --candidate A

    # Capture a single live frame from camera index 0 and inspect:
    python scripts/debug_preprocessing.py --camera 0 --candidate B

    # Run interactive live debug comparison window:
    python scripts/debug_preprocessing.py --live --camera 0
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.preprocessing import HandPreprocessor, DEFAULT_ROI_BBOX, clamp_roi
from recognition.predictor import GesturePredictor

def load_training_samples(class_name: str, count: int = 3) -> list:
    """Loads up to `count` training images from dataset/master/<class_name>/."""
    master_dir = PROJECT_ROOT / "dataset" / "master" / class_name
    samples = []
    if master_dir.exists():
        # Pick distributed sample files
        cand_files = [master_dir / f"{i}.jpg" for i in [1, 50, 100, 200, 500]]
        for cf in cand_files:
            if cf.exists():
                img = cv2.imread(str(cf), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples.append((cf.name, img))
            if len(samples) >= count:
                break
    return samples

def build_comparison_canvas(
    frame: np.ndarray,
    roi_bgr: np.ndarray,
    raw_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    model_input_50: np.ndarray,
    candidate_class: str,
    telemetry: dict,
    prediction: dict = None
) -> np.ndarray:
    """
    Constructs a professional, labeled multi-stage diagnostic dashboard canvas.
    Stages:
      1. Original frame with ROI box
      2. ROI crop (BGR)
      3. Raw skin segmentation mask
      4. Cleaned morphological mask (with contour)
      5. Final 50x50 model input (scaled 4x to 200x200)
      6. Candidate class training images (scaled 4x)
    """
    canvas_w = 1280
    canvas_h = 720
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)  # Sleek dark background

    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Header Banner ---
    cv2.rectangle(canvas, (0, 0), (canvas_w, 50), (35, 35, 42), -1)
    cv2.putText(canvas, "PREPROCESSING DIAGNOSTIC & TRAINING COMPARISON TOOL", (20, 34),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    status_str = "VALID HAND" if telemetry.get("hand_detected", False) else "NO HAND"
    status_color = (0, 220, 0) if telemetry.get("hand_detected", False) else (0, 80, 255)
    cv2.putText(canvas, f"STATUS: {status_str}", (850, 34), font, 0.75, status_color, 2, cv2.LINE_AA)

    # --- Top Row: 4 Panels (Stages 1 to 4) ---
    # Panel 1: Original Frame with ROI Box (300x225)
    f_display = frame.copy()
    rx, ry, rw, rh = telemetry.get("roi_bbox", DEFAULT_ROI_BBOX)
    box_color = (0, 255, 0) if telemetry.get("hand_detected") else (0, 200, 255)
    cv2.rectangle(f_display, (rx, ry), (rx + rw, ry + rh), box_color, 2)
    f_thumb = cv2.resize(f_display, (300, 225))
    canvas[80:305, 20:320] = f_thumb
    cv2.putText(canvas, "1. Camera Frame (ROI)", (20, 72), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel 2: ROI Crop (225x225)
    if roi_bgr is not None and roi_bgr.size > 0:
        roi_thumb = cv2.resize(roi_bgr, (225, 225))
    else:
        roi_thumb = np.zeros((225, 225, 3), dtype=np.uint8)
    canvas[80:305, 340:565] = roi_thumb
    cv2.putText(canvas, "2. ROI Crop", (340, 72), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel 3: Raw Segmentation Mask (225x225)
    if raw_mask is not None and raw_mask.size > 0:
        raw_bgr = cv2.cvtColor(cv2.resize(raw_mask, (225, 225), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    else:
        raw_bgr = np.zeros((225, 225, 3), dtype=np.uint8)
    canvas[80:305, 585:810] = raw_bgr
    cv2.putText(canvas, "3. Raw Skin Mask", (585, 72), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel 4: Cleaned Morphological Mask (225x225)
    if cleaned_mask is not None and cleaned_mask.size > 0:
        clean_bgr = cv2.cvtColor(cv2.resize(cleaned_mask, (225, 225), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        # Draw contour bbox if available
        c_bbox = telemetry.get("contour_bbox")
        if c_bbox:
            bx, by, bw, bh = c_bbox
            # Scale bbox to 225x225
            orig_h, orig_w = cleaned_mask.shape[:2]
            sx = 225.0 / orig_w if orig_w > 0 else 1.0
            sy = 225.0 / orig_h if orig_h > 0 else 1.0
            cv2.rectangle(clean_bgr, (int(bx * sx), int(by * sy)), (int((bx + bw) * sx), int((by + bh) * sy)), (0, 255, 0), 1)
    else:
        clean_bgr = np.zeros((225, 225, 3), dtype=np.uint8)
    canvas[80:305, 830:1055] = clean_bgr
    cv2.putText(canvas, "4. Morph Cleaned Mask", (830, 72), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Bottom Row: Model Input vs Real Training Images ---
    # Panel 5: Final 50x50 Model Input (scaled 4x to 200x200)
    if model_input_50 is not None and model_input_50.size > 0:
        inp_thumb = cv2.resize(model_input_50, (200, 200), interpolation=cv2.INTER_NEAREST)
        inp_bgr = cv2.cvtColor(inp_thumb, cv2.COLOR_GRAY2BGR)
    else:
        inp_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas[360:560, 20:220] = inp_bgr
    cv2.putText(canvas, "5. Model Input (50x50, 4x)", (20, 345), font, 0.5, (255, 220, 100), 1, cv2.LINE_AA)

    # Panels 6, 7, 8: Real Training Images from candidate class
    tr_samples = load_training_samples(candidate_class, count=3)
    offsets = [240, 460, 680]
    for idx, off in enumerate(offsets):
        cv2.putText(canvas, f"Training Sample {idx+1} ({candidate_class})", (off, 345), font, 0.45, (160, 220, 255), 1, cv2.LINE_AA)
        if idx < len(tr_samples):
            fname, s_img = tr_samples[idx]
            s_thumb = cv2.resize(s_img, (200, 200), interpolation=cv2.INTER_NEAREST)
            s_bgr = cv2.cvtColor(s_thumb, cv2.COLOR_GRAY2BGR)
            canvas[360:560, off:off+200] = s_bgr
            cv2.putText(canvas, fname, (off + 5, 555), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (off, 360), (off+200, 560), (45, 45, 55), 1)
            cv2.putText(canvas, "No sample found", (off + 25, 460), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # --- Telemetry & Diagnosis Panel (Bottom Right) ---
    t_box_x = 900
    t_box_y = 360
    cv2.rectangle(canvas, (t_box_x, t_box_y), (canvas_w - 20, canvas_h - 20), (35, 38, 48), -1)
    cv2.rectangle(canvas, (t_box_x, t_box_y), (canvas_w - 20, canvas_h - 20), (70, 75, 95), 1)
    cv2.putText(canvas, "DIAGNOSTIC METRICS", (t_box_x + 15, t_box_y + 25), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"Contour Area:   {telemetry.get('contour_area', 0.0):.0f} px",
        f"Bounding Box:   {telemetry.get('bbox_w', 0)} x {telemetry.get('bbox_h', 0)}",
        f"Aspect Ratio:   {telemetry.get('aspect_ratio', 0.0):.2f}",
        f"Foreground %:   {telemetry.get('fg_percent', 0.0):.1f}%",
        f"Extent:         {telemetry.get('extent', 0.0):.2f}",
        f"Result:         {telemetry.get('rejection_reason', 'N/A')}",
    ]
    if prediction:
        lines.append(f"Prediction:     {prediction.get('label', 'None')} ({prediction.get('confidence', 0.0)*100:.1f}%)")
    else:
        lines.append("Prediction:     [Inference Skipped]")

    y_pos = t_box_y + 55
    for l in lines:
        c = (220, 220, 220)
        if "Result:" in l:
            c = (100, 255, 100) if "Valid" in l else (100, 150, 255)
        elif "Prediction:" in l and prediction and prediction.get("confidence", 0) > 0.7:
            c = (255, 220, 100)
        cv2.putText(canvas, l, (t_box_x + 15, y_pos), font, 0.45, c, 1, cv2.LINE_AA)
        y_pos += 24

    # Bottom Instructions Bar
    cv2.putText(canvas, "Keys: [Q/ESC] Quit | [S] Save Stages | [C] Calibrate Skin | [SPACE] Next Frame",
                (20, canvas_h - 15), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return canvas

def run_diagnostic(
    image_path: str = None,
    camera_index: int = 0,
    candidate_class: str = "A",
    output_dir: str = "results/preprocessing_debug",
    save_stages: bool = False,
    live_mode: bool = False,
    roi_bbox: tuple = None
):
    """Executes the diagnostic pipeline and renders/saves the comparison."""
    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    roi = roi_bbox if roi_bbox else DEFAULT_ROI_BBOX
    preprocessor = HandPreprocessor(roi_bbox=roi)
    predictor = GesturePredictor()

    if image_path:
        # Static image mode
        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"[ERROR] Image not found: {p}")
            return False

        frame = cv2.imread(str(p))
        if frame is None:
            print(f"[ERROR] Failed to read image: {p}")
            return False

        print(f"Inspecting static image: {p.name} (shape: {frame.shape})")
        _process_and_save(frame, preprocessor, predictor, candidate_class, out_path, save_all=True)
        return True

    # Camera mode
    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {camera_index}.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    if not live_mode:
        # Grab a single frame
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame from camera.")
            return False
        print("Captured single camera frame. Processing stages...")
        _process_and_save(frame, preprocessor, predictor, candidate_class, out_path, save_all=save_stages or True)
        return True

    # Interactive live mode
    window_name = "Preprocessing Diagnostics"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Running interactive live mode. Press 'q' to quit, 's' to save stages, 'c' to calibrate.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_idx += 1
        canvas, prep_res, pred_res = _analyze_frame(frame, preprocessor, predictor, candidate_class)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            save_name = f"diag_frame_{frame_idx}"
            _save_intermediate_stages(out_path, save_name, frame, prep_res, pred_res, canvas)
            print(f"Saved diagnostic stages to {out_path / save_name}")
        elif key == ord('c'):
            print("Interactive skin calibration triggered...")
            # Interactive calibration from ROI center
            preprocessor.calibrate_from_frame(frame)

    cap.release()
    cv2.destroyAllWindows()
    return True

def _analyze_frame(frame, preprocessor, predictor, candidate_class):
    """Processes a frame through preprocessor and predictor, returning visual canvas."""
    res = preprocessor.process_frame(frame)
    telemetry = res["debug_info"]
    raw_mask = res.get("raw_mask", res.get("thresh_roi"))
    cleaned_mask = res.get("thresh_roi")
    model_input_50 = res.get("hand_crop_50")
    flipped = res.get("flipped_frame", frame)

    rx, ry, rw, rh = telemetry.get("roi_bbox", DEFAULT_ROI_BBOX)
    roi_bgr = flipped[ry:ry+rh, rx:rx+rw] if flipped is not None else None

    pred_res = None
    if res["hand_detected"]:
        pred_res = predictor.predict(res["processed_image"])

    canvas = build_comparison_canvas(
        frame=flipped,
        roi_bgr=roi_bgr,
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        model_input_50=model_input_50,
        candidate_class=candidate_class,
        telemetry=telemetry,
        prediction=pred_res
    )
    return canvas, res, pred_res

def _save_intermediate_stages(out_path, prefix, frame, prep_res, pred_res, canvas):
    """Saves all 5 stages and comparison canvas to disk."""
    save_dir = out_path / prefix
    save_dir.mkdir(parents=True, exist_ok=True)

    telemetry = prep_res["debug_info"]
    rx, ry, rw, rh = telemetry.get("roi_bbox", DEFAULT_ROI_BBOX)
    flipped = prep_res.get("flipped_frame", frame)
    roi_bgr = flipped[ry:ry+rh, rx:rx+rw] if flipped is not None else None

    # Stage 1: Original frame
    cv2.imwrite(str(save_dir / "1_original_frame.png"), flipped)
    # Stage 2: ROI Crop
    if roi_bgr is not None:
        cv2.imwrite(str(save_dir / "2_roi_crop.png"), roi_bgr)
    # Stage 3: Raw mask
    raw_mask = prep_res.get("raw_mask", prep_res.get("thresh_roi"))
    if raw_mask is not None:
        cv2.imwrite(str(save_dir / "3_raw_mask.png"), raw_mask)
    # Stage 4: Cleaned mask
    cleaned_mask = prep_res.get("thresh_roi")
    if cleaned_mask is not None:
        cv2.imwrite(str(save_dir / "4_cleaned_mask.png"), cleaned_mask)
    # Stage 5: 50x50 model input
    crop_50 = prep_res.get("hand_crop_50")
    if crop_50 is not None:
        cv2.imwrite(str(save_dir / "5_model_input_50x50.png"), crop_50)
        # Stage 6: Scaled up 200x200 (4x) nearest-neighbor preview
        enlarged_200 = cv2.resize(crop_50, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(save_dir / "6_model_input_enlarged.png"), enlarged_200)
        cv2.imwrite(str(save_dir / "5_model_input_enlarged.png"), enlarged_200)
    # Full diagnostic canvas
    cv2.imwrite(str(save_dir / "comparison_dashboard.png"), canvas)

    # Telemetry report text
    with open(save_dir / "telemetry_report.txt", "w") as f:
        f.write("=== PREPROCESSING DIAGNOSTIC REPORT ===\n")
        for k, v in telemetry.items():
            f.write(f"{k}: {v}\n")
        if pred_res:
            f.write(f"\nPrediction Class: {pred_res.get('class_id')}\n")
            f.write(f"Prediction Label: {pred_res.get('label')}\n")
            f.write(f"Confidence:       {pred_res.get('confidence')*100:.2f}%\n")
        else:
            f.write("\nInference Skipped (No Hand Detected)\n")

def _process_and_save(frame, preprocessor, predictor, candidate_class, out_path, save_all=True):
    """Processes single frame and prints telemetry."""
    canvas, prep_res, pred_res = _analyze_frame(frame, preprocessor, predictor, candidate_class)
    telemetry = prep_res["debug_info"]

    print("\n" + "=" * 65)
    print(" PREPROCESSING DIAGNOSTIC RESULT")
    print("=" * 65)
    print(f"Hand Detected:    {telemetry.get('hand_detected')}")
    print(f"Rejection Reason: {telemetry.get('rejection_reason')}")
    print(f"Contour Area:     {telemetry.get('contour_area', 0):.0f} px")
    print(f"Bounding Box:     {telemetry.get('bbox_w', 0)} x {telemetry.get('bbox_h', 0)}")
    print(f"Aspect Ratio:     {telemetry.get('aspect_ratio', 0):.2f}")
    print(f"Foreground %:     {telemetry.get('fg_percent', 0):.2f}%")
    print(f"Extent:           {telemetry.get('extent', 0):.2f}")
    if pred_res:
        print(f"Prediction:       {pred_res.get('label')} ({pred_res.get('confidence', 0)*100:.2f}%)")
    else:
        print("Prediction:       [No inference run - hand not detected]")
    print("=" * 65)

    if save_all:
        _save_intermediate_stages(out_path, "latest_inspection", frame, prep_res, pred_res, canvas)
        print(f"Diagnostic output saved to: {out_path / 'latest_inspection'}")

def main():
    parser = argparse.ArgumentParser(description="Sign Language Preprocessing Diagnostic Tool")
    parser.add_argument("--image", type=str, default="", help="Path to saved image to inspect (no camera required)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--candidate", type=str, default="A", help="Candidate class name for training image comparison (e.g. A, B, D, U, Fuck)")
    parser.add_argument("--outdir", type=str, default="results/preprocessing_debug", help="Directory to save diagnostic outputs")
    parser.add_argument("--save", action="store_true", help="Save intermediate pipeline stages to disk")
    parser.add_argument("--live", action="store_true", help="Launch interactive live webcam diagnostic window")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None, help="Custom ROI bbox")
    args = parser.parse_args()

    roi_bbox = tuple(args.roi) if args.roi else None
    run_diagnostic(
        image_path=args.image if args.image else None,
        camera_index=args.camera,
        candidate_class=args.candidate,
        output_dir=args.outdir,
        save_stages=args.save,
        live_mode=args.live,
        roi_bbox=roi_bbox
    )

if __name__ == "__main__":
    main()
