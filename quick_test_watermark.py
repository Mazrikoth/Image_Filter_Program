import os
import glob
import cv2
import torch
import config
import utils
from detector_ocr import BorderOCRDetector

# Optimize CPU threads
torch.set_num_threads(8)

def test_watermarks():
    print("==================================================================")
    print("          QUICK WATERMARK DETECTION TEST ON NEW SESSION           ")
    print("==================================================================")
    
    # Scan source directory for images
    image_patterns = [
        os.path.join(config.SOURCE_DIR, "*.png"),
        os.path.join(config.SOURCE_DIR, "*.jpg"),
        os.path.join(config.SOURCE_DIR, "*.jpeg")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
        
    print(f"Total images found: {len(image_paths)}")
    if not image_paths:
        print("No images found in the source directory.")
        return
        
    # Initialize EasyOCR detector
    detector = BorderOCRDetector()
    
    print("\nScanning first 50 images for watermarks...")
    detected_count = 0
    
    for idx, path in enumerate(image_paths[:50]):
        filename = os.path.basename(path)
        image_rgb = utils.load_image_rgb(path)
        if image_rgb is None:
            continue
            
        image_bgr = utils.rgb_to_bgr(image_rgb)
        
        has_watermark, details = detector.detect_watermark(image_bgr, threshold=config.WATERMARK_THRESHOLD)
        if has_watermark:
            detected_count += 1
            print(f"[{idx+1}/50] [!] DETECTED in {filename}:")
            for detail in details:
                print(f"    - {detail}")
        else:
            # Print periodic clean images just to show progress
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[{idx+1}/50] Scan progress... Clean: {filename}")
                
    print("\n==================================================================")
    print(f"TEST RUN COMPLETE: Detected watermarks in {detected_count} out of 50 images.")
    print("==================================================================")

if __name__ == "__main__":
    test_watermarks()
