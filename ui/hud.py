"""
ui/hud.py
---------
OpenCV-based Heads-Up Display (HUD) and Dashboard renderer.
Draws side-by-side interactive camera feed and dashboard containing gesture
recognition metrics, confidence meters, text/calculator outputs, and hotkey guides.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

class HUDRenderer:
    """
    Renders real-time HUD with webcam overlay and information dashboard.
    """
    def __init__(self, frame_width: int = 800, frame_height: int = 600, panel_width: int = 400, demo_safe: bool = True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = panel_width
        self.total_width = frame_width + panel_width  # 800 + 400 = 1200
        self.demo_safe = demo_safe

        # Color Palette (BGR)
        self.COLOR_BG = (22, 22, 26)              # Dark slate background
        self.COLOR_PANEL_HEADER = (35, 35, 42)    # Subtle card header
        self.COLOR_ACCENT = (255, 178, 50)        # Gold/Cyan accent
        self.COLOR_WHITE = (245, 245, 245)
        self.COLOR_GRAY = (140, 140, 150)
        self.COLOR_GREEN = (60, 220, 100)         # Stable / Match
        self.COLOR_YELLOW = (40, 210, 255)        # Detecting
        self.COLOR_RED = (70, 70, 230)            # Unknown / Error
        self.COLOR_BLUE = (240, 140, 40)          # Mode indicator

    def render(
        self,
        flipped_frame: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        hand_detected: bool,
        hand_crop_50: Optional[np.ndarray],
        prediction: Dict[str, Any],
        stabilizer_state: Dict[str, Any],
        controller_state: Dict[str, Any],
        fps: float,
        debug: bool = False,
        debug_info: Optional[Dict[str, Any]] = None,
        last_commit_event: Optional[str] = None
    ) -> np.ndarray:
        """
        Assembles the complete HUD view: 800x600 camera view + 400x600 dashboard (1200x600 total).
        Includes enlarged model input preview, instructional 400x400 ROI UX, and optional debug telemetry.
        """
        # 1. Prepare Camera Frame
        cam_view = flipped_frame.copy()
        if cam_view.shape[:2] != (self.frame_height, self.frame_width):
            cam_view = cv2.resize(cam_view, (self.frame_width, self.frame_height))

        # Draw ROI Box & Clear UX Instructions
        rx, ry, rw, rh = roi_bbox
        roi_color = self.COLOR_GREEN if hand_detected else self.COLOR_YELLOW
        cv2.rectangle(cam_view, (rx, ry), (rx + rw, ry + rh), roi_color, 2)

        # Draw Corner Accents on ROI Box
        corner_len = 28
        # Top-Left
        cv2.line(cam_view, (rx, ry), (rx + corner_len, ry), roi_color, 4)
        cv2.line(cam_view, (rx, ry), (rx, ry + corner_len), roi_color, 4)
        # Top-Right
        cv2.line(cam_view, (rx + rw, ry), (rx + rw - corner_len, ry), roi_color, 4)
        cv2.line(cam_view, (rx + rw, ry), (rx + rw, ry + corner_len), roi_color, 4)
        # Bottom-Left
        cv2.line(cam_view, (rx, ry + rh), (rx + corner_len, ry + rh), roi_color, 4)
        cv2.line(cam_view, (rx, ry + rh), (rx, ry + rh - corner_len), roi_color, 4)
        # Bottom-Right
        cv2.line(cam_view, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), roi_color, 4)
        cv2.line(cam_view, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), roi_color, 4)

        # Clear Instruction Banner Above ROI
        if hand_detected:
            # Green "Hand Detected" pill
            badge_w = min(190, rw)
            cv2.rectangle(cam_view, (rx, max(0, ry - 38)), (rx + badge_w, ry - 6), (18, 55, 24), -1)
            cv2.rectangle(cam_view, (rx, max(0, ry - 38)), (rx + badge_w, ry - 6), self.COLOR_GREEN, 1)
            cv2.putText(
                cam_view,
                "Hand Detected",
                (rx + 10, ry - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                self.COLOR_GREEN,
                2
            )
        else:
            # Amber "Place your hand inside this box" instructional banner
            cv2.rectangle(cam_view, (rx, max(0, ry - 38)), (rx + rw, ry - 6), (20, 22, 28), -1)
            cv2.rectangle(cam_view, (rx, max(0, ry - 38)), (rx + rw, ry - 6), self.COLOR_YELLOW, 1)
            cv2.putText(
                cam_view,
                "Place your hand inside this box",
                (rx + 12, ry - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                self.COLOR_WHITE,
                2
            )
            cv2.putText(
                cam_view,
                "[No Hand Detected]",
                (rx + 14, ry + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.COLOR_GRAY,
                1
            )
            # Palm calibration target guide
            pcx = rx + (rw - 80) // 2
            pcy = ry + (rh - 80) // 2
            cv2.rectangle(cam_view, (pcx, pcy), (pcx + 80, pcy + 80), (80, 85, 95), 1)
            cv2.putText(cam_view, "Palm [K]", (pcx + 10, pcy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 155, 170), 1)

        # Draw Model Input Preview (200x200, exact 4x nearest-neighbor)
        thumb_size = 200
        tx, ty = 20, 45
        cv2.putText(cam_view, "MODEL INPUT 50x50", (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_ACCENT, 1, cv2.LINE_AA)

        if hand_detected and hand_crop_50 is not None and hand_crop_50.size > 0:
            thumb = cv2.resize(hand_crop_50, (thumb_size, thumb_size), interpolation=cv2.INTER_NEAREST)
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(thumb_bgr, (0, 0), (thumb_size - 1, thumb_size - 1), self.COLOR_GREEN, 2)
            cam_view[ty:ty + thumb_size, tx:tx + thumb_size] = thumb_bgr
        else:
            # Clearly show that CNN is receiving NO input when hand is absent
            cv2.rectangle(cam_view, (tx, ty), (tx + thumb_size, ty + thumb_size), (24, 24, 30), -1)
            cv2.rectangle(cam_view, (tx, ty), (tx + thumb_size, ty + thumb_size), (60, 60, 75), 1)
            cv2.putText(cam_view, "No Hand", (tx + 50, ty + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.COLOR_RED, 2)
            cv2.putText(cam_view, "No CNN Input", (tx + 40, ty + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GRAY, 1)

        # Draw FPS on Camera View
        cv2.putText(
            cam_view,
            f"FPS: {fps:.1f}",
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            self.COLOR_GREEN,
            2
        )

        # Draw Debug Mode Overlay (Telemetry Box) positioned below 200x200 preview
        if debug and debug_info:
            card_x, card_y, card_w, card_h = 20, 255, 310, 335
            overlay = cam_view.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (14, 14, 18), -1)
            cv2.addWeighted(overlay, 0.88, cam_view, 0.12, 0, cam_view)
            cv2.rectangle(cam_view, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_ACCENT, 1)

            cv2.putText(cam_view, "DEBUG TELEMETRY", (card_x + 10, card_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_ACCENT, 2)

            d_rx, d_ry, d_rw, d_rh = debug_info.get("roi_bbox", (rx, ry, rw, rh))
            area = debug_info.get("contour_area", 0.0)
            bw = debug_info.get("bbox_w", 0)
            bh = debug_info.get("bbox_h", 0)
            ar = debug_info.get("aspect_ratio", 0.0)
            fg = debug_info.get("fg_percent", 0.0)
            d_hand = debug_info.get("hand_detected", False)
            reason = debug_info.get("rejection_reason", "")
            is_demo_safe = controller_state.get("demo_safe", self.demo_safe)
            raw_cid = prediction.get("class_id", -1)
            raw_lbl = prediction.get("label", "None")
            raw_conf = prediction.get("confidence", 0.0) * 100.0
            st_lbl = stabilizer_state.get("stable_label", "None")
            commit_str = last_commit_event if last_commit_event else "None"

            if is_demo_safe:
                if raw_lbl.lower() == "fuck" or raw_cid == 42 or prediction.get("is_excluded", False):
                    raw_lbl = "[Excluded in demo]"
                    raw_cid = -1
                if "fuck" in str(st_lbl).lower():
                    st_lbl = "Gesture unavailable in demo mode"
                if "fuck" in str(commit_str).lower():
                    commit_str = "[Excluded in demo]"

            lines = [
                (f"ROI: ({d_rx}, {d_ry}, {d_rw}x{d_rh})", self.COLOR_WHITE),
                (f"Area: {area:.0f} px", self.COLOR_WHITE),
                (f"BBox: {bw}x{bh} px", self.COLOR_WHITE),
                (f"Aspect Ratio: {ar:.2f}", self.COLOR_WHITE),
                (f"FG Percent: {fg:.1f}%", self.COLOR_WHITE),
                (f"Hand Detect: {'YES' if d_hand else 'NO'}", self.COLOR_GREEN if d_hand else self.COLOR_RED),
                (f"Status: {reason[:26]}", self.COLOR_YELLOW if not d_hand else self.COLOR_GREEN),
                (f"Raw Pred: [{raw_cid}] {raw_lbl}", self.COLOR_WHITE),
                (f"Raw Conf: {raw_conf:.1f}%", self.COLOR_WHITE),
                (f"Stable: {st_lbl}", self.COLOR_YELLOW),
                (f"Commit: {commit_str[:24]}", self.COLOR_ACCENT)
            ]

            ly = card_y + 42
            for ltext, lcol in lines:
                cv2.putText(cam_view, ltext, (card_x + 10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, lcol, 1)
                ly += 24

        # 2. Build Dashboard Panel (400x600)
        panel = np.zeros((self.frame_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = self.COLOR_BG

        # Title Header
        cv2.rectangle(panel, (0, 0), (self.panel_width, 55), self.COLOR_PANEL_HEADER, -1)
        cv2.putText(
            panel,
            "SIGN LANGUAGE INTERPRETER",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            self.COLOR_ACCENT,
            2
        )

        # Mode Badge & Demo Safe Indicator Badge
        is_demo_safe = controller_state.get("demo_safe", self.demo_safe)
        mode_str = controller_state.get("mode", "TEXT")
        badge_bg = (180, 100, 20) if mode_str == "TEXT" else (160, 50, 120)
        cv2.rectangle(panel, (20, 72), (205, 107), badge_bg, -1)
        cv2.putText(
            panel,
            f"MODE: {mode_str}",
            (28, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            self.COLOR_WHITE,
            2
        )

        if is_demo_safe:
            cv2.rectangle(panel, (220, 72), (380, 107), (35, 110, 45), -1)
            cv2.rectangle(panel, (220, 72), (380, 107), self.COLOR_GREEN, 1)
            cv2.putText(
                panel,
                "DEMO SAFE",
                (236, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                self.COLOR_WHITE,
                2
            )
        else:
            cv2.rectangle(panel, (220, 72), (380, 107), (30, 30, 120), -1)
            cv2.rectangle(panel, (220, 72), (380, 107), self.COLOR_RED, 1)
            cv2.putText(
                panel,
                "DEV (RAW)",
                (242, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_WHITE,
                2
            )

        # Gesture & Confidence Section
        label = prediction.get("display_label", "Unknown")
        if is_demo_safe:
            if "fuck" in str(label).lower() or prediction.get("is_excluded", False) or prediction.get("class_id", -1) == 42:
                label = "Gesture unavailable in demo mode"

        conf = prediction.get("confidence", 0.0) * 100.0
        status = stabilizer_state.get("status", "SEARCHING")
        if is_demo_safe and "fuck" in str(status).lower():
            status = "UNAVAILABLE (DEMO SAFE)"

        cv2.putText(panel, "Current Gesture:", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.COLOR_GRAY, 1)

        # Large Gesture Label
        if label == "Gesture unavailable in demo mode":
            cv2.putText(panel, "Gesture unavailable", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.78, self.COLOR_YELLOW, 2)
            cv2.putText(panel, "in demo mode", (20, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_GRAY, 1)
            label_color = self.COLOR_YELLOW
        else:
            label_color = self.COLOR_GREEN if prediction.get("is_confident", False) else self.COLOR_RED
            cv2.putText(panel, f"{label}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.45, label_color, 3)

        # Confidence Bar
        cv2.putText(panel, f"Confidence: {conf:.1f}%", (20, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1)
        bar_x, bar_y, bar_w, bar_h = 20, 248, 360, 12
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_w = int((conf / 100.0) * bar_w)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), label_color, -1)

        # Status Tag
        cv2.putText(panel, f"Status: {status}", (20, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_YELLOW, 1)

        # Divider
        cv2.line(panel, (20, 300), (self.panel_width - 20, 300), (50, 50, 60), 1)

        # Mode Specific Content
        if mode_str == "TEXT":
            cv2.putText(panel, "Accumulated Sentence:", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.COLOR_GRAY, 1)
            sentence = controller_state.get("sentence", "")
            if is_demo_safe and sentence:
                import re
                sentence = re.sub(r'(?i)\bfuck\b', '', sentence).strip()
            self._draw_multiline_text(panel, sentence if sentence else "[Empty Sentence]", (20, 365), 360)

        elif mode_str == "CALCULATOR":
            cv2.putText(panel, "Calculator Expression:", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.COLOR_GRAY, 1)
            expr = controller_state.get("expression", "")
            cv2.putText(panel, f"{expr if expr else '0'}", (20, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.90, self.COLOR_WHITE, 2)

            res = controller_state.get("calc_result", "")
            if res:
                res_color = self.COLOR_RED if controller_state.get("calc_has_error", False) else self.COLOR_GREEN
                cv2.putText(panel, f"= {res}", (20, 415), cv2.FONT_HERSHEY_SIMPLEX, 1.05, res_color, 2)
            else:
                cv2.putText(panel, "Gesture 36 / '=' to compute", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GRAY, 1)

        # Bottom Instructions Footer (anchored to bottom of 600px panel)
        footer_h = 70
        footer_y = self.frame_height - footer_h
        cv2.rectangle(panel, (0, footer_y), (self.panel_width, self.frame_height), (15, 15, 18), -1)
        if mode_str == "TEXT":
            cv2.putText(panel, "[Q] Quit  [C] Clear  [S] Speak  [M] Mode", (20, footer_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_WHITE, 1)
            cv2.putText(panel, "[K] Calibrate  [D] Save Frame  [Del] Back", (20, footer_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GRAY, 1)
        else:
            cv2.putText(panel, "[Q] Quit  [C] Clear  [M] Mode  [36] Calc", (20, footer_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_WHITE, 1)
            cv2.putText(panel, "[K] Calibrate  [D] Save Frame  [=] Solve", (20, footer_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GRAY, 1)

        # 3. Concatenate side-by-side: 800x600 + 400x600 = 1200x600
        canvas = np.hstack((cam_view, panel))
        return canvas

    def _draw_multiline_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], max_width: int):
        """Draws wrapped text inside bounding width."""
        x, y = pos
        words = text.split(" ")
        lines = []
        current_line = ""

        for w in words:
            test_line = current_line + (" " if current_line else "") + w
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            if tw < max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = w
        if current_line:
            lines.append(current_line)

        # Allow up to 5 lines on the larger panel
        if len(lines) > 5:
            lines = lines[-5:]

        for line in lines:
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_WHITE, 2)
            y += 28
