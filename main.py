import os
import glob
import sys
import torch
import config
import utils
from detector_ocr import BorderOCRDetector
from detector_hands import HandAnatomyDetector

# Optimize PyTorch CPU execution
torch.set_num_threads(8)

def run_sorting():
    watermark_only = "--watermark-only" in sys.argv
    
    print("==================================================================")
    print("              STARTING IMAGE FILTER SORTING RUN                   ")
    if watermark_only:
        print("                   (WATERMARK ONLY MODE)                         ")
    print("==================================================================")
    
    # 1. Ensure target directories exist
    dirs_to_create = [config.WATERMARKED_DIR]
    if not watermark_only:
        dirs_to_create.extend([config.DEFORMED_DIR, config.COMBINED_DIR])
    utils.setup_directories(dirs_to_create)
    
    # 2. Scan source directory for images (excluding subdirectories)
    image_patterns = [
        os.path.join(config.SOURCE_DIR, "*.png"),
        os.path.join(config.SOURCE_DIR, "*.jpg"),
        os.path.join(config.SOURCE_DIR, "*.jpeg")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
        
    print(f"Found {len(image_paths)} images in source directory: {config.SOURCE_DIR}")
    if not image_paths:
        print("No images found to process. Exiting.")
        return
        
    # 3. Initialize detectors
    ocr_detector = BorderOCRDetector()
    hand_detector = None if watermark_only else HandAnatomyDetector()
    
    print("\nProcessing images...")
    processed_count = 0
    watermarked_count = 0
    deformed_count = 0
    combined_count = 0
    clean_count = 0
    
    for path in image_paths:
        filename = os.path.basename(path)
        processed_count += 1
        print(f"[{processed_count}/{len(image_paths)}] Processing {filename}...", end="", flush=True)
        
        image_rgb = utils.load_image_rgb(path)
        if image_rgb is None:
            print(" -> FAILED TO LOAD (DELETING CORRUPT IMAGE)")
            try:
                os.remove(path)
            except Exception as del_err:
                print(f"    Failed to delete corrupt image {filename}: {del_err}")
            continue
            
        image_bgr = utils.rgb_to_bgr(image_rgb)
        
        # Detect watermark
        has_watermark, ocr_details = ocr_detector.detect_watermark(image_bgr, threshold=config.WATERMARK_THRESHOLD)
        
        # Detect deformed hands
        if watermark_only:
            has_deformed = False
            hand_details = []
        else:
            has_deformed, hand_details = hand_detector.detect_bad_hands(image_rgb, threshold=config.HAND_DEFORMITY_THRESHOLD)
        
        # Routing logic
        status_str = ""
        if has_watermark and has_deformed:
            utils.move_file(path, config.COMBINED_DIR)
            combined_count += 1
            status_str = f" -> WATERMARKED & DEFORMED ({', '.join(ocr_details + hand_details)})"
        elif has_watermark:
            utils.move_file(path, config.WATERMARKED_DIR)
            watermarked_count += 1
            status_str = f" -> WATERMARKED ({', '.join(ocr_details)})"
        elif has_deformed:
            utils.move_file(path, config.DEFORMED_DIR)
            deformed_count += 1
            status_str = f" -> DEFORMED ({', '.join(hand_details)})"
        else:
            clean_count += 1
            status_str = " -> CLEAN"
            
        print(status_str)
        
    print("\n==================================================================")
    print("                       SORTING RUN COMPLETED                      ")
    print("==================================================================")
    print(f"Total processed: {processed_count}")
    print(f"  - Watermarked only: {watermarked_count}")
    if not watermark_only:
        print(f"  - Deformed hands only: {deformed_count}")
        print(f"  - Watermarked & Deformed: {combined_count}")
    print(f"  - Clean (remained in source): {clean_count}")
    print("==================================================================")

if __name__ == "__main__":
    run_sorting()
