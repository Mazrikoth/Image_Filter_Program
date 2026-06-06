import os
import shutil
import cv2
import numpy as np
from PIL import Image

def setup_directories(dirs):
    """Ensure target folders exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def load_image_rgb(img_path):
    """
    Robust image loader that bypasses standard OpenCV file loading issues
    by loading via PIL and converting to a numpy RGB array.
    """
    try:
        with Image.open(img_path) as pil_img:
            pil_img.load()  # Force loading to check for truncated or corrupt streams
            converted_img = pil_img.convert('RGB')
            image_rgb = np.array(converted_img)
            return image_rgb
    except Exception as e:
        print(f"Error loading image {os.path.basename(img_path)}: {e}")
        return None

def rgb_to_bgr(image_rgb):
    """Convert RGB array to BGR format for OpenCV operations."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def move_file(source_path, target_dir):
    """Safely move a file to target folder, handling conflicts by overwrite."""
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_dir, filename)
    try:
        shutil.move(source_path, target_path)
    except Exception as e:
        print(f"Error moving {filename} to {target_dir}: {e}")
