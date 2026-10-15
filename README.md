# Unified Sign Language Interpreter

A real-time American Sign Language (ASL) and gesture recognition system powered by TensorFlow/Keras and OpenCV. This project unifies, modernizes, and builds upon research and codebases from:

1. **EvilPort2 (`Sign-Language` by Dibakar Saha):** Primary dataset and core deep learning pipeline reference.
2. **harshbg (`Sign-Language-Interpreter-using-Deep-Learning` by Harsh Gupta et al., HackUNT 2019):** Secondary reference for hackathon demos, presentation materials, and skin histogram calibrations.

---

## 1. Project Overview & Vocabulary

The project supports **44 canonical gesture classes** (IDs `0` through `43`), covering the complete ASL fingerspelling alphabet, digits 0–9, and high-frequency communicative gestures:

- **Alphabets (26 classes):** `A` through `Z` (IDs `0` to `25`)
- **Digits (10 classes):** `0` through `9` (IDs `26` to `35`)
- **Special Gestures & Words (8 classes):**
  - `36`: **Best of Luck** (Thumbs-up / Good luck; also serves as calculate/confirm trigger in calculator mode and space in text mode)
  - `37`: **You** (Index pointing outward)
  - `38`: **I/Me** (Index pointing toward speaker)
  - `39`: **Like** (Thumbs up)
  - `40`: **Remember** (Forehead/temple touch)
  - `41`: **Love** (Crossed hands sign)
  - `42`: **Fuck** (Middle finger gesture — strictly preserved as Class 42)
  - `43`: **I love you** (Classic ASL ILY sign: thumb, index, pinky extended)

---

## 2. Dataset Architecture

- **Total Images:** 105,600 images (exactly 2,400 images per class across 44 classes)
- **Image Specifications:** 50x50 pixels, 1-channel Grayscale (8-bit), `.jpg` format
- **Composition per Class:** 1,200 captured webcam frames + 1,200 horizontally flipped augmented frames
- **Storage Strategy:** Zero data duplication using NTFS directory junctions on Windows (or symlinks on POSIX). Both original upstream repositories remain 100% untouched.

### Group-Safe Splits (Zero Train/Test Leakage)
Because images `1`–`1200` and `1201`–`2400` are horizontal flip pairs, every pair shares the same `group_id` (`1`–`1200`). Splits are partitioned strictly by `(class_id, group_id)`:
- **Train Split (70%):** 73,920 images (1,680 images per class)
- **Validation Split (15%):** 15,840 images (360 images per class)
- **Test Split (15%):** 15,840 images (360 images per class)
- **Leakage Verification:** 0 overlapping pairs between Train, Validation, and Test sets.

---

## 3. Primary Model & Performance Benchmark

- **Primary Model:** `models/diagnostic_modern_cnn.keras`
- **Offline Benchmark Performance on 15,840 Test Images:**
  - **Test Accuracy:** **99.89%**
  - **Macro F1 Score:** **99.89%**
  - **Weighted F1 Score:** **99.89%**
  - **Minimum Per-Class F1:** $\ge 97.58\%$ across all 44 classes (Class 42 "Fuck" and Class 43 "I love you" both achieved **1.0000** F1).

### Model Architecture
```text
Input: (50, 50, 1) Grayscale
  ├── Rescaling(1./255)
  ├── Conv2D(32, 3x3, same) + BatchNormalization + ReLU + MaxPooling2D(2x2)
  ├── Conv2D(64, 3x3, same) + BatchNormalization + ReLU + MaxPooling2D(2x2)
  ├── Conv2D(128, 3x3, same) + BatchNormalization + ReLU + MaxPooling2D(2x2)
  ├── Flatten
  ├── Dense(256) + BatchNormalization + ReLU + Dropout(0.3)
  └── Dense(44, Softmax)
```

> [!IMPORTANT]
> **Real-World Performance Notice:**  
> **99.89% is the offline dataset benchmark.** Real-world webcam accuracy may differ because of lighting conditions, background clutter, skin tone variance, hand segmentation quality, camera angle, pose, and motion blur.

---

## 4. Installation & Setup

Ensure Python 3.10 or 3.11 is installed.

```bash
# 1. Clone or navigate to the project directory
cd Unified-Sign-Language

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 5. How to Run

### Real-Time Live Webcam Application

```bash
# Default mode (Text mode, camera index 0):
python app.py

# Launch in Calculator Mode:
python app.py --mode calculator

# Specify custom camera index or confidence threshold:
python app.py --camera 1 --threshold 0.75
```

### Command-Line Single-Image Prediction & Batch Test

```bash
# Predict a single image:
python scripts/predict.py --image dataset/master/Fuck/1.jpg

# Predict with Top-5 candidates:
python scripts/predict.py --image dataset/master/A/1.jpg --top-k 5

# Verify on 10 random test images:
python scripts/predict.py --verify-random 10
```

### Run Unit Tests & Automated Pipeline Verification

```bash
# Run complete test suite (23 unit tests):
python -m pytest

# Test the webcam & inference pipeline without requiring a physical camera:
python scripts/test_webcam_pipeline.py
```

---

## 6. Application Modes & Controls

The real-time application features an interactive dual-panel OpenCV Heads-Up Display (HUD) with live webcam monitoring, hand ROI extraction, threshold thumbnail, confidence meter, and output text.

### Text Mode
- **Fingerspelling:** Extending letters `A`–`Z` appends the character to the active sentence.
- **Digits:** Gestures `0`–`9` append the corresponding digit.
- **Words / Space:**
  - Gesture `36` ("Best of Luck"): Inserts a space (`" "`).
  - Gestures `37`–`43`: Insert vocabulary words (`" You "`, `" I "`, `" like "`, `" remember "`, `" love "`, `" fuck "`, `" I love you "`).

### Calculator Mode
- **Digits `0`–`9`:** Formulate expressions using sign language digits.
- **Operators:** Entered via keyboard (`+`, `-`, `*`, `/`, `.`).
- **Calculate:** Perform Gesture `36` ("Best of Luck") or press `=` / `Enter` to solve.
- **Safety:** Powered by an AST-based mathematical evaluator (no arbitrary `eval()`).

### Keyboard Controls
| Key | Function | Mode |
| :--- | :--- | :--- |
| `Q` / `ESC` | Quit Application | Both |
| `M` | Toggle Mode (TEXT &harr; CALCULATOR) | Both |
| `C` | Clear sentence or expression | Both |
| `S` | Speak sentence via Text-to-Speech (pyttsx3) | Both |
| `Backspace` | Delete last character | Both |
| `+`, `-`, `*`, `/`, `.` | Enter math operators | Calculator |
| `=`, `Enter` | Calculate expression | Calculator |

---

## 7. Known Limitations & Best Practices

1. **Illumination:** The OpenCV skin segmentation works best under consistent, bright lighting. Dim lighting or strong backlighting may degrade hand contours.
2. **Background:** Avoid complex skin-colored backgrounds behind the green ROI box. A plain, non-reflective wall provides optimal segmentation.
3. **Hand Placement:** Position hand comfortably inside the green bounding box on screen. The hand indicator will switch from yellow ("Hand ROI") to green ("Hand Detected") when valid contours are detected.
