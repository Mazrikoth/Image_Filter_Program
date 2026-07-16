import os

# Central Paths
SOURCE_DIR = r"c:\Users\DjLog\OneDrive - The RadioCave Non-Profit Corporation\Downloads\midjourney_session_1"
WATERMARKED_DIR = os.path.join(SOURCE_DIR, "watermarked")
DEFORMED_DIR = os.path.join(SOURCE_DIR, "deformed_hands")
COMBINED_DIR = os.path.join(SOURCE_DIR, "watermarked_and_deformed")
CORRUPTED_DIR = os.path.join(SOURCE_DIR, "corrupted")
MULTI_HEAD_DIR = os.path.join(SOURCE_DIR, "multi_headed")

# Central Detection Thresholds (Autotuned by calibration.py)
WATERMARK_THRESHOLD = 0.15             # Calibrated OCR text probability (tightened to catch 20% missed)
HAND_DEFORMITY_THRESHOLD = 0.55        # Calibrated BARC bad-anatomy crop probability
WHOLE_IMAGE_DEFORMED_THRESHOLD = 0.45  # Standard fallback
DRAGON_HEAD_THRESHOLD = 0.15           # Confidence threshold for OWL-ViT dragon head detection

# Multiprocessing limits
NUM_WORKERS = 16  # Spawn 16 parallel worker processes (fast processing)
