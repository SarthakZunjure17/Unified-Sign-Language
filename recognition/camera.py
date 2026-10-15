"""
recognition/camera.py
---------------------
Real-time webcam pipeline orchestrating video capture, preprocessing,
CNN inference, stabilization, controller actions, and HUD rendering.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2

from .preprocessing import HandPreprocessor, DEFAULT_ROI_BBOX, clamp_roi
from .predictor import GesturePredictor
from .stabilizer import TemporalStabilizer
from application.controller import AppController
from ui.hud import HUDRenderer

class CameraPipeline:
    """
    Main runtime loop connecting camera input to recognition, application, and UI.
    Standardized to 800x600 camera feed and 1200x600 complete UI canvas.
    """
    def __init__(
        self,
        camera_index: int = 0,
        mode: str = "TEXT",
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.70,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        debug: bool = False,
        demo_safe: bool = True
    ):
        self.camera_index = camera_index
        self.debug = debug
        self.demo_safe = demo_safe
        if roi_bbox is not None:
            actual_roi = clamp_roi(roi_bbox, 800, 600)
        else:
            actual_roi = DEFAULT_ROI_BBOX

        self.preprocessor = HandPreprocessor(roi_bbox=actual_roi)
        self.predictor = GesturePredictor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            demo_safe=demo_safe
        )
        self.stabilizer = TemporalStabilizer(
            min_confidence=confidence_threshold,
            demo_safe=demo_safe
        )
        self.controller = AppController(
            default_mode=mode,
            demo_safe=demo_safe
        )
        self.hud = HUDRenderer(
            frame_width=800,
            frame_height=600,
            panel_width=400,
            demo_safe=demo_safe
        )
        self.last_commit_info: Optional[str] = None
        self.frame_count: int = 0

    def save_debug_snapshot(self, prep_res: Dict[str, Any], pred_res: Dict[str, Any]) -> str:
        """
        Saves a live example of all 6 pipeline stages:
        1. Original frame
        2. ROI crop
        3. Raw segmentation mask
        4. Cleaned mask
        5. Final 50x50 model input
        6. Enlarged 200x200 preview
        to the dedicated debug directory (debug_frames/).
        """
        import json
        project_root = Path(__file__).resolve().parent.parent
        debug_dir = project_root / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        sub_dir = debug_dir / f"snapshot_{timestamp_str}_f{self.frame_count}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        flipped = prep_res.get("flipped_frame")
        if flipped is not None:
            cv2.imwrite(str(sub_dir / "1_original_frame.png"), flipped)

        rx, ry, rw, rh = prep_res.get("roi_bbox", (360, 100, 400, 400))
        if flipped is not None:
            roi_crop = flipped[ry:ry + rh, rx:rx + rw]
            cv2.imwrite(str(sub_dir / "2_roi_crop.png"), roi_crop)

        raw_mask = prep_res.get("raw_mask")
        if raw_mask is not None:
            cv2.imwrite(str(sub_dir / "3_raw_mask.png"), raw_mask)

        cleaned_mask = prep_res.get("thresh_roi")
        if cleaned_mask is not None:
            cv2.imwrite(str(sub_dir / "4_cleaned_mask.png"), cleaned_mask)

        hand_crop_50 = prep_res.get("hand_crop_50")
        if hand_crop_50 is not None:
            cv2.imwrite(str(sub_dir / "5_model_input_50x50.png"), hand_crop_50)
            enlarged = cv2.resize(hand_crop_50, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(sub_dir / "6_model_input_enlarged.png"), enlarged)

        telemetry = prep_res.get("debug_info", {})
        meta = {
            "frame_number": self.frame_count,
            "timestamp": timestamp_str,
            "hand_detected": prep_res.get("hand_detected", False),
            "prediction": {
                "class_id": pred_res.get("class_id", -1),
                "label": pred_res.get("label", "None"),
                "confidence": float(pred_res.get("confidence", 0.0)),
                "display_label": pred_res.get("display_label", "Unknown")
            },
            "telemetry": telemetry
        }
        with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return str(sub_dir)

    def run(self):
        """Starts real-time video capture loop."""
        print(f"Opening webcam (Index {self.camera_index})...", flush=True)
        cap = cv2.VideoCapture(self.camera_index)

        # Handle camera open failure gracefully
        if not cap.isOpened():
            print(f"\n[ERROR] Unable to open video source at camera index {self.camera_index}.", flush=True)
            print("Please ensure a webcam is connected and accessible, or specify a different camera index with --camera <index>.", flush=True)
            return False

        # Attempt to configure resolution to 800x600 (4:3 aspect ratio)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        window_name = "Unified Sign Language Interpreter"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        prev_time = time.time()
        fps = 30.0

        print("\nApplication running. Press 'q' or 'ESC' in the display window to exit.", flush=True)
        if self.demo_safe:
            print("[POLICY] Demo Safe Mode: ACTIVE (Class 42 excluded from UI, text, audio, and logs)", flush=True)
        else:
            print("[POLICY] Unrestricted Dev Mode: ACTIVE (Raw 44 classes)", flush=True)
        if self.debug:
            print("[DEBUG] Verbose debug HUD overlay and metrics are ENABLED.", flush=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[WARNING] Failed to grab frame from camera stream.")
                    time.sleep(0.05)
                    continue

                # Standardize frame to 800x600 (maintaining 4:3 camera aspect ratio)
                if frame.shape[1] != 800 or frame.shape[0] != 600:
                    frame = cv2.resize(frame, (800, 600))

                self.frame_count += 1
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # 1. Preprocess frame
                prep_res = self.preprocessor.process_frame(frame)
                hand_detected = prep_res["hand_detected"]
                proc_img = prep_res["processed_image"]

                # 2. Model Inference
                if hand_detected:
                    pred_res = self.predictor.predict(proc_img)
                else:
                    pred_res = {
                        "class_id": -1,
                        "label": "None",
                        "display_label": "No Hand",
                        "confidence": 0.0,
                        "is_confident": False,
                        "top_k": []
                    }

                # 3. Temporal Stabilization
                stab_res = self.stabilizer.update(pred_res, hand_detected=hand_detected)

                # 4. Gesture Commit Dispatch
                if stab_res["is_committed"]:
                    commit_lbl = stab_res["committed_label"]
                    commit_cid = stab_res["committed_class"]
                    if self.demo_safe and (commit_cid == 42 or str(commit_lbl).lower() == "fuck"):
                        pass
                    else:
                        self.last_commit_info = f"'{commit_lbl}' (f#{self.frame_count})"
                        if self.debug:
                            print(f"[DEBUG COMMIT] Frame {self.frame_count}: Committed '{commit_lbl}' (Class {commit_cid})")
                        self.controller.handle_committed_gesture(
                            commit_cid,
                            commit_lbl
                        )

                # 5. Render HUD
                hud_view = self.hud.render(
                    flipped_frame=prep_res["flipped_frame"],
                    roi_bbox=prep_res["roi_bbox"],
                    hand_detected=hand_detected,
                    hand_crop_50=prep_res.get("hand_crop_50"),
                    prediction=pred_res,
                    stabilizer_state=stab_res,
                    controller_state=self.controller.get_display_state(),
                    fps=fps,
                    debug=self.debug,
                    debug_info=prep_res.get("debug_info"),
                    last_commit_event=self.last_commit_info
                )

                # 6. Display frame
                cv2.imshow(window_name, hud_view)

                # 7. Keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key in [ord('k'), ord('K')]:
                        calib_res = self.preprocessor.calibrate_from_frame(frame)
                        print(f"\n[CALIBRATION] {calib_res['message']}", flush=True)
                        self.last_commit_info = "Palm Calibrated"
                    elif key in [ord('d'), ord('D')]:
                        saved_path = self.save_debug_snapshot(prep_res, pred_res)
                        print(f"\n[DEBUG SNAPSHOT] Saved all 6 pipeline stages to: {saved_path}", flush=True)
                        self.last_commit_info = "Debug Frame Saved"
                        self.controller.status_message = "Debug Frame Saved"
                    else:
                        keep_running = self.controller.handle_key(key)
                        if not keep_running:
                            print("Exiting application...")
                            break

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and display windows closed.")

        return True
