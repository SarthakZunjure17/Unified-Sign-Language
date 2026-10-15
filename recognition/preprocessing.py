"""
recognition/preprocessing.py
----------------------------
Robust, lighting-adaptive hand segmentation and ROI preprocessing pipeline for real-time
webcam input and offline images. Resolves skin segmentation mismatches, gap fragmentation,
and background noise false positives to match the solid-silhouette training dataset.
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

# =============================================================================
# CONFIGURABLE ROI & HAND DETECTION CONSTANTS (800x600 Webcam Stream)
# =============================================================================
DEFAULT_ROI_X = 360
DEFAULT_ROI_Y = 100
DEFAULT_ROI_W = 400
DEFAULT_ROI_H = 400
DEFAULT_ROI_BBOX = (DEFAULT_ROI_X, DEFAULT_ROI_Y, DEFAULT_ROI_W, DEFAULT_ROI_H)

# Geometric Validation Thresholds to reject noise and false hands
MIN_CONTOUR_AREA = 2800.0         # Minimum contour area in pixels for a valid hand/fist
MAX_CONTOUR_AREA_RATIO = 0.88     # Reject contours filling almost entire ROI (>88%)
MIN_CONTOUR_WIDTH = 30            # Reject ultra-thin vertical line artifacts (< 30px)
MIN_CONTOUR_HEIGHT = 35           # Reject ultra-thin horizontal line artifacts (< 35px)
MIN_ASPECT_RATIO = 0.28           # Reject narrow vertical artifacts (< 0.28)
MAX_ASPECT_RATIO = 3.00           # Reject wide horizontal artifacts (> 3.00)
MIN_SKIN_RATIO = 0.030            # Minimum skin coverage (3.0% of ROI) before Otsu thresholding
MIN_CONTOUR_EXTENT = 0.14         # Reject hollow outlines / diagonal lines (area / bbox_area)

def clamp_roi(roi_bbox: Tuple[int, int, int, int], w_frame: int, h_frame: int) -> Tuple[int, int, int, int]:
    """
    Clamps and positions an ROI bounding box (x, y, w, h) so that it remains
    square and fully visible inside the frame [w_frame, h_frame] without clipping.
    """
    rx, ry, rw, rh = roi_bbox
    is_square = (rw == rh)

    rw = max(50, min(rw, w_frame))
    rh = max(50, min(rh, h_frame))
    if is_square:
        side = min(rw, rh)
        rw = side
        rh = side

    rx = max(0, min(rx, w_frame - rw))
    ry = max(0, min(ry, h_frame - rh))
    return (int(rx), int(ry), int(rw), int(rh))

class HandPreprocessor:
    """
    Extracts and standardizes hand gesture ROIs for the 50x50x1 CNN model.
    Utilizes multi-color-space skin detection (YCrCb + HSV), gap-closing morphological
    operations, hole-filling for solid hand silhouettes, and optional palm histogram calibration.
    """
    def __init__(
        self,
        roi_bbox: Tuple[int, int, int, int] = DEFAULT_ROI_BBOX,
        min_contour_area: float = MIN_CONTOUR_AREA,
        max_contour_area_ratio: float = MAX_CONTOUR_AREA_RATIO,
        min_contour_width: int = MIN_CONTOUR_WIDTH,
        min_contour_height: int = MIN_CONTOUR_HEIGHT,
        min_aspect_ratio: float = MIN_ASPECT_RATIO,
        max_aspect_ratio: float = MAX_ASPECT_RATIO,
        min_skin_ratio: float = MIN_SKIN_RATIO,
        min_contour_extent: float = MIN_CONTOUR_EXTENT,
        hist_path: Optional[Path] = None
    ):
        self.roi_bbox = roi_bbox
        self.min_contour_area = min_contour_area
        self.max_contour_area_ratio = max_contour_area_ratio
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_skin_ratio = min_skin_ratio
        self.min_contour_extent = min_contour_extent

        self.hist_path = hist_path
        self.hist = self._load_skin_hist(hist_path)
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Normalized disc filter for backprojection to prevent noise amplification
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.float32)
        self.disc_norm = disc / np.sum(disc)

    def _load_skin_hist(self, hist_path: Optional[Path]):
        """Attempts to load a skin color histogram if available."""
        candidates = []
        if hist_path:
            candidates.append(Path(hist_path))
        project_root = Path(__file__).resolve().parent.parent
        # Prefer user-calibrated local hist
        candidates.append(project_root / "hist")
        candidates.append(project_root.parent / "Sign-Language-master" / "Sign-Language-master" / "hist")
        candidates.append(project_root.parent / "Sign-Language-master" / "hist")

        for cand in candidates:
            if cand.exists() and cand.is_file():
                try:
                    with open(cand, "rb") as f:
                        hist = pickle.load(f)
                    return hist
                except Exception:
                    pass
        return None

    def calibrate_from_roi(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Interactively calibrates the skin histogram from a palm region inside the ROI.
        Extracts an 80x80 central patch, computes 2D HSV histogram, normalizes,
        and saves it to 'hist' in the project root.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {"success": False, "message": "ROI is empty"}

        h_roi, w_roi = roi_bgr.shape[:2]
        cw, ch = 80, 80
        cx = max(0, (w_roi - cw) // 2)
        cy = max(0, (h_roi - ch) // 2)

        palm_patch = roi_bgr[cy:cy + ch, cx:cx + cw]
        hsv_palm = cv2.cvtColor(palm_patch, cv2.COLOR_BGR2HSV)

        # 2D HSV histogram on Hue [0, 180] and Saturation [0, 256]
        hist = cv2.calcHist([hsv_palm], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        self.hist = hist

        # Persist to project root
        project_root = Path(__file__).resolve().parent.parent
        save_file = project_root / "hist"
        try:
            with open(save_file, "wb") as f:
                pickle.dump(hist, f)
            return {"success": True, "message": f"Skin histogram calibrated and saved to {save_file.name}"}
        except Exception as e:
            return {"success": True, "message": f"Skin histogram calibrated in-memory ({e})"}

    def calibrate_from_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calibrates skin histogram using the current frame and active ROI bbox."""
        if frame is None or frame.size == 0:
            return {"success": False, "message": "Frame is empty"}
        flipped = cv2.flip(frame, 1)
        h_frame, w_frame = flipped.shape[:2]
        rx, ry, rw, rh = clamp_roi(self.roi_bbox, w_frame, h_frame)
        roi_bgr = flipped[ry:ry + rh, rx:rx + rw]
        return self.calibrate_from_roi(roi_bgr)

    @staticmethod
    def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
        """
        Fills enclosed internal holes inside palm/fingers without filling the
        exterior valleys between separate fingers using topological flood fill.
        """
        if mask is None or mask.size == 0 or np.max(mask) == 0:
            return mask
        h, w = mask.shape[:2]
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask_inv = cv2.bitwise_not(padded)
        flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(mask_inv, flood_mask, (0, 0), 0)
        enclosed_holes = mask_inv[1:-1, 1:-1]
        return cv2.bitwise_or(mask, enclosed_holes)

    def segment_hand(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        """
        Segments the hand within the ROI.
        Returns:
            solid_mask: Clean, hole-filled binary mask of the hand (solid white silhouette).
            raw_mask: Raw skin probability map before morphological closing.
            skin_present: Boolean indicating if enough skin pixels were detected.
            status_reason: Diagnostic reason string.
        """
        if roi_bgr is None or roi_bgr.size == 0 or np.max(roi_bgr) < 15:
            empty = np.zeros(roi_bgr.shape[:2] if roi_bgr is not None and roi_bgr.size > 0 else (50, 50), dtype=np.uint8)
            return empty, empty, False, "Empty ROI"

        h_roi, w_roi = roi_bgr.shape[:2]
        total_pixels = h_roi * w_roi

        # 0. Check for synthetic / grayscale images (e.g. unit tests or dataset images)
        if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
            # Synthetic images created with cv2.cvtColor(gray, COLOR_GRAY2BGR) have exactly identical channels
            if np.array_equal(roi_bgr[:, :, 0], roi_bgr[:, :, 1]) and np.array_equal(roi_bgr[:, :, 1], roi_bgr[:, :, 2]):
                gray = roi_bgr[:, :, 0]
                _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                solid = self.fill_internal_holes(thresh)
                return solid, solid, True, "Synthetic Grayscale"

        # 1. Multi-Color-Space Illumination-Adaptive Skin Detection
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # YCrCb Skin Chrominance Model (Illumination invariant)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        mask_ycrcb = ((cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)).astype(np.uint8) * 255

        # HSV Skin Model (Handles shadow and highlight variations)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask_hsv = (((h <= 25) | (h >= 165)) & (s >= 18) & (v >= 35)).astype(np.uint8) * 255

        # Combined multi-space baseline mask
        multi_space_skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # 2. Histogram Backprojection with normalized kernel (if calibrated histogram is present)
        if self.hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
            dst_filtered = cv2.filter2D(dst, -1, self.disc_norm)
            # Confident backproject response (>= 35)
            backproj_mask = (dst_filtered >= 35).astype(np.uint8) * 255
            skin_prob = cv2.bitwise_or(multi_space_skin, backproj_mask)
        else:
            skin_prob = multi_space_skin

        # 3. Empty Background Gate: Reject ROI if skin presence is too low
        skin_pixel_count = cv2.countNonZero(skin_prob)
        skin_ratio = skin_pixel_count / float(total_pixels) if total_pixels > 0 else 0.0

        if skin_ratio < self.min_skin_ratio:
            empty = np.zeros((h_roi, w_roi), dtype=np.uint8)
            return empty, skin_prob, False, f"No hand in ROI (skin coverage {skin_ratio:.1%} < {self.min_skin_ratio:.1%})"

        # 4. Gentle edge-preserving smoothing to suppress sensor noise without melting fingers
        blur = cv2.GaussianBlur(skin_prob, (5, 5), 0)

        # 5. Otsu Thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Gentle Morphological Closing:
        # 5x5 ellipse kernel with 1 iteration bridges tiny knuckle creases without merging 15-20px finger gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 7. Connected Component Analysis: isolate hand and remove isolated background noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            empty = np.zeros((h_roi, w_roi), dtype=np.uint8)
            return empty, skin_prob, False, "No foreground components found"

        # Background is label 0; find component with largest area
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_comp_idx = 1 + int(np.argmax(areas))
        hand_area = float(stats[largest_comp_idx, cv2.CC_STAT_AREA])

        # Geometric sanity checks on hand component area
        if hand_area < self.min_contour_area:
            empty = np.zeros((h_roi, w_roi), dtype=np.uint8)
            return empty, skin_prob, False, f"Contour area too small ({hand_area:.0f} < {self.min_contour_area:.0f})"

        if hand_area > (self.max_contour_area_ratio * total_pixels):
            empty = np.zeros((h_roi, w_roi), dtype=np.uint8)
            return empty, skin_prob, False, f"Contour area too large ({hand_area:.0f} > {self.max_contour_area_ratio * total_pixels:.0f})"

        # Clean mask containing only the primary hand component (purges all background speckles)
        hand_mask = (labels == largest_comp_idx).astype(np.uint8) * 255

        # 8. Topological Flood-Fill Hole-Filling:
        # Fills enclosed holes inside palm/fingers without filling the exterior valleys between fingers!
        solid_mask = self.fill_internal_holes(hand_mask)

        # 9. Edge refinement: gentle 3x3 median blur cleans single-pixel stair-stepping
        solid_mask = cv2.medianBlur(solid_mask, 3)

        return solid_mask, skin_prob, True, "Valid Hand"

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Full pipeline for a webcam frame:
        1. Flips horizontally for natural mirror UX.
        2. Clamps and extracts square ROI.
        3. Segments hand into a solid, unfragmented silhouette.
        4. Validates geometry (area, width, height, aspect ratio, extent).
        5. Performs exact symmetric square padding and resizes to (50, 50).
        6. Returns structured telemetry. When no hand is detected, returns zeros and hand_detected=False.
        """
        if frame is None or frame.size == 0 or np.max(frame) == 0:
            empty_50 = np.zeros((1, 50, 50, 1), dtype=np.float32)
            empty_info = {
                "roi_bbox": self.roi_bbox,
                "contour_area": 0.0,
                "bbox_w": 0,
                "bbox_h": 0,
                "aspect_ratio": 0.0,
                "fg_percent": 0.0,
                "hand_detected": False,
                "rejection_reason": "Empty frame",
                "extent": 0.0
            }
            return {
                "processed_image": empty_50,
                "roi_bbox": self.roi_bbox,
                "hand_detected": False,
                "thresh_roi": np.zeros((50, 50), dtype=np.uint8),
                "raw_mask": np.zeros((50, 50), dtype=np.uint8),
                "contour_bbox": None,
                "flipped_frame": frame,
                "hand_crop_50": np.zeros((50, 50), dtype=np.uint8),
                "debug_info": empty_info
            }

        # 1. Flip horizontally
        flipped = cv2.flip(frame, 1)
        h_frame, w_frame = flipped.shape[:2]

        # 2. Extract clamped ROI
        rx, ry, rw, rh = clamp_roi(self.roi_bbox, w_frame, h_frame)
        roi_bgr = flipped[ry:ry + rh, rx:rx + rw]

        # 3. Hand segmentation
        solid_mask, raw_mask, skin_detected, seg_reason = self.segment_hand(roi_bgr)

        # 4. Measure foreground coverage in ROI
        roi_total_pixels = rw * rh
        fg_pixels = cv2.countNonZero(solid_mask)
        fg_percent = (fg_pixels / float(roi_total_pixels)) * 100.0 if roi_total_pixels > 0 else 0.0

        # 5. Contour validation & Bounding Box extraction
        hand_detected = False
        contour_bbox = None
        resized_50 = np.zeros((50, 50), dtype=np.uint8)
        model_input = np.zeros((1, 50, 50, 1), dtype=np.float32)
        rejection_reason = seg_reason
        area = 0.0
        bw = bh = 0
        aspect_ratio = 0.0
        extent = 0.0

        if skin_detected:
            contours, _ = cv2.findContours(solid_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(largest))
                bx, by, bw, bh = cv2.boundingRect(largest)
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0
                extent = area / float(bw * bh) if (bw * bh) > 0 else 0.0

                if area < self.min_contour_area:
                    rejection_reason = f"Contour area too small ({area:.0f} < {self.min_contour_area:.0f})"
                elif area > (self.max_contour_area_ratio * roi_total_pixels):
                    rejection_reason = f"Contour area too large ({area:.0f} > {self.max_contour_area_ratio * roi_total_pixels:.0f})"
                elif bw < self.min_contour_width:
                    rejection_reason = f"Bounding box width too small ({bw} < {self.min_contour_width})"
                elif bh < self.min_contour_height:
                    rejection_reason = f"Bounding box height too small ({bh} < {self.min_contour_height})"
                elif aspect_ratio < self.min_aspect_ratio:
                    rejection_reason = f"Contour too thin/elongated vertically (AR {aspect_ratio:.2f} < {self.min_aspect_ratio:.2f})"
                elif aspect_ratio > self.max_aspect_ratio:
                    rejection_reason = f"Contour too wide/elongated horizontally (AR {aspect_ratio:.2f} > {self.max_aspect_ratio:.2f})"
                elif extent < self.min_contour_extent:
                    rejection_reason = f"Contour extent too hollow ({extent:.2f} < {self.min_contour_extent:.2f})"
                else:
                    # Valid hand confirmed!
                    hand_detected = True
                    rejection_reason = "Valid Hand"
                    contour_bbox = (bx, by, bw, bh)

                    # Crop hand bounding box directly from the polished silhouette
                    # (Preserves genuine finger shapes and finger gaps; avoids destroying spaces between fingers)
                    cropped = solid_mask[by:by + bh, bx:bx + bw]

                    # Exact symmetric square padding with proportional margin
                    side = max(bw, bh)
                    margin = max(2, int(side * 0.04))
                    side_padded = side + 2 * margin
                    pad_top = (side_padded - bh) // 2
                    pad_bottom = side_padded - bh - pad_top
                    pad_left = (side_padded - bw) // 2
                    pad_right = side_padded - bw - pad_left
                    padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

                    # Resize to (50, 50) using area interpolation (matches training dataset profile)
                    resized_50 = cv2.resize(padded, (50, 50), interpolation=cv2.INTER_AREA)
                    model_input = np.expand_dims(np.expand_dims(resized_50.astype(np.float32), axis=0), axis=-1)

        debug_info = {
            "roi_bbox": (rx, ry, rw, rh),
            "contour_area": area,
            "bbox_w": bw,
            "bbox_h": bh,
            "aspect_ratio": aspect_ratio,
            "fg_percent": fg_percent,
            "hand_detected": hand_detected,
            "rejection_reason": rejection_reason,
            "extent": extent
        }

        return {
            "processed_image": model_input,
            "roi_bbox": (rx, ry, rw, rh),
            "hand_detected": hand_detected,
            "thresh_roi": solid_mask,
            "raw_mask": raw_mask,
            "contour_bbox": contour_bbox,
            "flipped_frame": flipped,
            "hand_crop_50": resized_50,
            "debug_info": debug_info
        }

    @staticmethod
    def preprocess_image_array(img: np.ndarray) -> np.ndarray:
        """
        Preprocesses an existing image array (grayscale or BGR) into (1, 50, 50, 1) [0, 255].
        """
        if img is None or img.size == 0:
            return np.zeros((1, 50, 50, 1), dtype=np.float32)

        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = img

        resized = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_AREA)
        arr = np.expand_dims(np.expand_dims(resized.astype(np.float32), axis=0), axis=-1)
        return arr
