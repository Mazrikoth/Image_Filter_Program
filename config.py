import os

# Central Paths
SOURCE_DIR = r"C:\Downloads\midjourney_session_1"
WATERMARKED_DIR = os.path.join(SOURCE_DIR, "watermarked")
DEFORMED_DIR = os.path.join(SOURCE_DIR, "deformed_hands")
COMBINED_DIR = os.path.join(SOURCE_DIR, "watermarked_and_deformed")

# Central Detection Thresholds (Autotuned by calibration.py)
WATERMARK_THRESHOLD = 0.40             # Calibrated OCR text probability
HAND_DEFORMITY_THRESHOLD = 0.60        # Calibrated BARC bad-anatomy crop probability
WHOLE_IMAGE_DEFORMED_THRESHOLD = 0.45  # Standard fallback
