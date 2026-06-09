import os

# Central Paths
SOURCE_DIR = r"C:\Downloads\midjourney_session_1"
WATERMARKED_DIR = os.path.join(SOURCE_DIR, "watermarked")
DEFORMED_DIR = os.path.join(SOURCE_DIR, "deformed_hands")
COMBINED_DIR = os.path.join(SOURCE_DIR, "watermarked_and_deformed")
CORRUPTED_DIR = os.path.join(SOURCE_DIR, "corrupted")

# Central Detection Thresholds (Autotuned by calibration.py)
WATERMARK_THRESHOLD = 0.20             # Calibrated OCR text probability (tightened to catch 20% missed)
HAND_DEFORMITY_THRESHOLD = 0.60        # Calibrated BARC bad-anatomy crop probability
WHOLE_IMAGE_DEFORMED_THRESHOLD = 0.45  # Standard fallback
