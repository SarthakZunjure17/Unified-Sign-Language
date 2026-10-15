#!/usr/bin/env python3
"""
app.py
------
Unified Real-Time American Sign Language Interpreter.
Executes the live webcam stream, hand ROI segmentation, 99.89% CNN inference,
temporal stabilization, Text / Calculator controllers, speech output, and OpenCV HUD.

Usage:
    python app.py
    python app.py --mode text
    python app.py --mode calculator
    python app.py --camera 0
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.camera import CameraPipeline

def main():
    parser = argparse.ArgumentParser(description="Unified Real-Time Sign Language Interpreter")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--mode", type=str, choices=["text", "calculator"], default="text", help="Initial mode: 'text' or 'calculator'")
    parser.add_argument("--model", type=str, default="", help="Path to trained .keras model")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug HUD overlay and metrics")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                        help="Custom ROI bbox (X Y W H), e.g. --roi 200 40 400 400")
    parser.add_argument("--demo-safe", dest="demo_safe", action="store_true", default=True,
                        help="Enable classroom safe demo mode (default: True, excludes Class 42 'Fuck')")
    parser.add_argument("--unrestricted", dest="demo_safe", action="store_false",
                        help="DEVELOPMENT ONLY: Disable demo safety policy and expose raw 44 classes")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    roi_bbox = tuple(args.roi) if args.roi else None

    print("=" * 65, flush=True)
    print(" UNIFIED SIGN LANGUAGE INTERPRETER - REAL-TIME APP", flush=True)
    print("=" * 65, flush=True)
    print(f"Initial Mode:         {args.mode.upper()}", flush=True)
    print(f"Safety Policy:        {'DEMO SAFE (Class 42 Excluded)' if args.demo_safe else 'DEVELOPMENT (RAW UNRESTRICTED)'}", flush=True)
    print(f"Webcam Index:         {args.camera}", flush=True)
    print(f"Confidence Threshold: {args.threshold:.2f}", flush=True)
    print(f"Debug Mode:           {'ENABLED' if args.debug else 'DISABLED'}", flush=True)
    print(f"UI Dimensions:        800x600 Camera + 400x600 Dashboard (1200x600)", flush=True)
    if roi_bbox:
        from recognition.preprocessing import clamp_roi
        clamped = clamp_roi(roi_bbox, 800, 600)
        if clamped != roi_bbox:
            print(f"Custom ROI:           {roi_bbox} (Clamped to {clamped})", flush=True)
        else:
            print(f"Custom ROI:           {roi_bbox}", flush=True)
    else:
        print(f"Webcam Hand ROI:      400x400 (DEFAULT: 360, 100, 400, 400)", flush=True)
    print("=" * 65, flush=True)

    pipeline = CameraPipeline(
        camera_index=args.camera,
        mode=args.mode,
        model_path=model_path,
        confidence_threshold=args.threshold,
        roi_bbox=roi_bbox,
        debug=args.debug,
        demo_safe=args.demo_safe
    )

    success = pipeline.run()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
