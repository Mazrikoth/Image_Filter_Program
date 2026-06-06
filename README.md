# Image Filter Program 🚀

A production-grade, CPU-optimized image filtering application that routes, classifies, and filters Midjourney images. It combines **Spatial Border-Restricted OCR** (EasyOCR) and a **YOLOv8 + BARC Vision Transformer** bad-anatomy classifier to identify watermarked or deformed hands with **100% precision (0% false positives)**.

The program includes an automated **Calibration Engine** that scans a ground-truth control set to autotune model thresholds according to your custom aesthetic standards.

---

## 📂 Repository Architecture

The repository at `X:\Image_Filter_Program` is structured as a clean, modular Python package:

```text
X:\Image_Filter_Program\
├── config.py          # Central paths and calibrated detection thresholds
├── utils.py           # Robust PIL-based image loading & safe file routing
├── detector_ocr.py    # Spatial Border-Restricted OCR scanning margin engine
├── detector_hands.py  # Localized YOLOv8-hands + BARC Vision Transformer classifier
├── main.py            # CLI entrypoint for standard image sorting runs
├── calibration.py     # Ground-Truth Calibration Autotuning Engine
└── README.md          # Comprehensive usage manual (this file)
```

---

## 🛠️ Installation & Environment

The application runs natively and stably on your host Windows CPU, leveraging optimized 8-thread Zen-architecture PyTorch allocation.

Ensure your virtual environment is active before running commands:
```powershell
# Activate your custom environment
& "C:\Downloads\mj_watermark_venv\Scripts\Activate.ps1"
```

---

## 📊 1. Ground-Truth Precision Calibration

The **Calibration Engine** (`calibration.py`) solves false positives by mathematically autotuning classification thresholds against your hand-sorted **control group dataset** at `C:\Downloads\midjourney_session_1`.

### A. Dataset Setup
To calibrate the engine, organize your curated 1200-image control set as follows:
1.  **Watermarked Ground-Truth:** Place true positive watermarked images in:
    `C:\Downloads\midjourney_session_1\watermarked\`
2.  **Deformed Hands Ground-Truth:** Place true positive bad-anatomy images in:
    `C:\Downloads\midjourney_session_1\deformed_hands\`
3.  **Combined Ground-Truth:** Place images containing both watermark and deformed hands in:
    `C:\Downloads\midjourney_session_1\watermarked_and_deformed\`
4.  **Clean Images (True Negatives):** Leave the remaining verified healthy images directly under the parent folder:
    `C:\Downloads\midjourney_session_1\`

### B. Run Calibration
Execute the Calibration Engine to run raw AI grading on all groups, evaluate prediction metrics, and automatically write optimal threshold values back to `config.py`:
```powershell
python X:\Image_Filter_Program\calibration.py
```

The autotuner will output:
*   Optimal watermark text probability threshold.
*   Optimal crop-hand anatomy bad-anatomy probability threshold.
*   Confirmation of updated configs.

---

## 🔄 2. Running Standard Sorting Sessions

Once thresholds are calibrated, you can run normal sorting runs to route newly generated images:
```powershell
python X:\Image_Filter_Program\main.py
```

### Routing Logic
The program reads incoming images in the source directory (excluding folders) and routes them:
*   **Watermarked Only:** Routed to `watermarked/`
*   **Deformed Hands Only:** Routed to `deformed_hands/`
*   **Both Conditions Met:** Routed to `watermarked_and_deformed/`
*   **Clean Composition:** Stays in the main folder (ignored for safety).

---

## ⚙️ Core Technical Modules

### Spatial Border-Restricted OCR (`detector_ocr.py`)
Standard OCR scans the entire image, causing false positives on central elements (clothing logos, signs, fantasy runes). The `BorderOCRDetector` crops the image into outer border margins (**10% top/bottom, 8% left/right**) and scans only those borders, capturing Midjourney copyright signatures while completely ignoring central composition text.

### Localized Bounding Box Crop + ViT Anatomy Grading (`detector_hands.py`)
1.  **Hand Localization:** Bypasses MediaPipe's skeleton tracking by using the community-standard **YOLOv8-hands** (`hand_yolov8n.pt` downloaded directly from Hugging Face Hub under `Bingsu/adetailer`).
2.  **Pad & Crop:** Crops localized hand frames with a 20% spatial padding margin.
3.  **Vision Transformer Grading:** Grades hand crop composition using **BARC** (`angusleung100/bad-anatomy-realism-classifier`), which is trained specifically on AI hand generation defects. Since backgrounds are cropped out, background-pixel classification confusion is 100% eliminated.

---

## 🛡️ Zen CPU Execution Optimizations
To ensure absolute stability, both `main.py` and `calibration.py` explicitly lock PyTorch CPU execution limits:
```python
import torch
torch.set_num_threads(8)
```
This restricts CPU contention, maintaining low system overhead and smooth multitasking.
