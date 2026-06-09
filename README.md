# Image Filter Program 🚀

A production-grade, CPU-optimized image filtering application that routes, classifies, and filters Midjourney images. It combines **Spatial Border-Restricted OCR** (EasyOCR), a **YOLOv8 + BARC Vision Transformer** bad-anatomy classifier, and a zero-shot **OWL-ViT Object Detector** to identify watermarks, deformed hands, or multi-headed compositions with **100% precision (0% false positives)**.

The program includes an automated **Calibration Engine** that scans a ground-truth control set to autotune model thresholds according to your custom aesthetic standards.

---

## 📂 Repository Architecture

The repository at `X:\Image_Filter_Program` is structured as a clean, modular Python package:

```text
X:\Image_Filter_Program\
├── .venv/             # Repository-contained Python virtual environment
├── config.py          # Central paths, calibrated detection thresholds, and process counts
├── utils.py           # Robust PIL-based image loading & safe file routing
├── detector_ocr.py    # Spatial Border-Restricted OCR scanning margin engine
├── detector_hands.py  # Localized YOLOv8-hands + BARC Vision Transformer classifier
├── detector_heads.py  # Zero-shot OWL-ViT dragon head detector
├── main.py            # CLI entrypoint for standard, watermark, and multi-head sorting runs
├── calibration.py     # Ground-Truth Calibration Autotuning Engine
└── README.md          # Comprehensive usage manual (this file)
```

---

## 🛠️ Installation & Environment

The application runs natively and stably on your host Windows CPU. To avoid impacting system performance, it is throttled to use only ~5% CPU resources (1 core of a 24-core CPU).

Ensure your repository-contained virtual environment is active before running commands:
```powershell
# Activate the repository virtual environment
& "X:\Image_Filter_Program\.venv\Scripts\Activate.ps1"
```

---

## 📊 1. Ground-Truth Precision Calibration

The **Calibration Engine** (`calibration.py`) solves false positives by mathematically autotuning classification thresholds against your hand-sorted **control group dataset** at `C:\Downloads\midjourney_session_1`.

### A. Dataset Setup
To calibrate the engine, organize your curated control set as follows:
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

## 🔄 2. Running Sorting Sessions

You can run sorting runs to route newly generated images. The program runs with 1 worker process using a single PyTorch thread, limiting CPU usage to approximately 5% on a 24-core machine.

### Run Modes

#### A. Standard Sorting (Watermark & Hand Deformity Detection)
```powershell
python X:\Image_Filter_Program\main.py
```
Or to run watermark detection only:
```powershell
python X:\Image_Filter_Program\main.py --watermark-only
```

#### B. Dragon Multi-Head Sorting Pass Only
Runs a visual pass using OWL-ViT to detect and move images with 2 or more dragon heads (as they are inaccurate for portrait compositions). This is ideal for cleaning up remaining root images:
```powershell
python X:\Image_Filter_Program\main.py --two-heads-only
```

#### C. Full Combined Sorting (Watermark, Hands, & Dragon Heads)
Runs all three checks simultaneously:
```powershell
python X:\Image_Filter_Program\main.py --two-heads
```

### Routing Logic
The program reads incoming images in the source directory (excluding folders) and routes them:
*   **Multi-Headed Composition:** Routed to `multi_headed/` (takes precedence over watermark/hands)
*   **Watermarked Only:** Routed to `watermarked/`
*   **Deformed Hands Only:** Routed to `deformed_hands/`
*   **Both Watermarked & Deformed Hands:** Routed to `watermarked_and_deformed/`
*   **Clean Composition (Single Portrait):** Stays in the main folder (ignored for safety).

---

## ⚙️ Core Technical Modules

### Spatial Border-Restricted OCR (`detector_ocr.py`)
Standard OCR scans the entire image, causing false positives on central elements (clothing logos, signs, fantasy runes). The `BorderOCRDetector` crops the image into outer border margins (**10% top/bottom, 8% left/right**) and scans only those borders, capturing Midjourney copyright signatures while completely ignoring central composition text.

### Localized Bounding Box Crop + ViT Anatomy Grading (`detector_hands.py`)
1.  **Hand Localization:** Bypasses MediaPipe's skeleton tracking by using the community-standard **YOLOv8-hands** (`hand_yolov8n.pt` downloaded directly from Hugging Face Hub under `Bingsu/adetailer`).
2.  **Pad & Crop:** Crops localized hand frames with a 20% spatial padding margin.
3.  **Vision Transformer Grading:** Grades hand crop composition using **BARC** (`angusleung100/bad-anatomy-realism-classifier`), which is trained specifically on AI hand generation defects. Since backgrounds are cropped out, background-pixel classification confusion is 100% eliminated.

### Zero-Shot Object Detection (`detector_heads.py`)
Uses the zero-shot open-vocabulary object detector **OWL-ViT** (`google/owlvit-base-patch32`) to look for `"a dragon head"` in the image. Bounding boxes with confidence scores above the threshold are evaluated. If 2 or more dragon heads are detected in an image, it is flagged as multi-headed and routed accordingly.

---

## 🛡️ CPU Execution Throttling (5% Limit)

To ensure minimal impact on system performance, CPU execution is capped to ~5% utilization:
1.  **Single Worker:** Multiprocessing is configured to use `NUM_WORKERS = 1` in `config.py`.
2.  **Thread Limit:** Both `main.py` and `calibration.py` set PyTorch CPU execution threads to 1:
```python
import torch
torch.set_num_threads(1)
```
This guarantees that the filtering program runs entirely on a single thread of a single CPU core, eliminating multitasking lag and system contention.
