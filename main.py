import os
import glob
import sys
import torch
import config
import utils
import concurrent.futures

# Optimize PyTorch global CPU execution to 1 thread for main process
torch.set_num_threads(1)

# Global variables in the worker process
ocr_detector = None
hand_detector = None
dragon_detector = None

def init_worker(watermark_only, two_heads_only, two_heads):
    """
    Runs once per worker process to initialize the detectors and configure CPU threads.
    """
    global ocr_detector, hand_detector, dragon_detector
    
    # Restrict PyTorch thread count inside each worker to 1 to prevent thrashing/deadlocks.
    torch.set_num_threads(1)
    
    # Initialize only what is needed for the active run mode to save memory/startup time
    if two_heads_only:
        from detector_heads import DragonHeadDetector
        dragon_detector = DragonHeadDetector()
        return

    from detector_ocr import BorderOCRDetector
    ocr_detector = BorderOCRDetector()
    
    if not watermark_only:
        from detector_hands import HandAnatomyDetector
        hand_detector = HandAnatomyDetector()
        
    if two_heads:
        from detector_heads import DragonHeadDetector
        dragon_detector = DragonHeadDetector()

def process_image_worker(path, watermark_only, two_heads_only, two_heads, watermark_threshold, hand_threshold, dragon_threshold):
    """
    Runs model inference inside the worker process.
    """
    global ocr_detector, hand_detector, dragon_detector
    import utils
    import os
    
    filename = os.path.basename(path)
    image_rgb = utils.load_image_rgb(path)
    if image_rgb is None:
        return path, "CORRUPT", "FAILED TO LOAD (MOVING TO CORRUPTED FOLDER)", filename
        
    # Mode 1: Dragon heads visual pass only
    if two_heads_only:
        has_multi, count, details = dragon_detector.detect_dragon_heads(image_rgb, threshold=dragon_threshold)
        if has_multi:
            status = "MULTI_HEAD"
            details_str = f"MULTI_HEAD ({count} heads detected: {', '.join(details)})"
        else:
            status = "CLEAN"
            details_str = "CLEAN"
        return path, status, details_str, filename

    # Mode 2: Standard/Watermark runs
    image_bgr = utils.rgb_to_bgr(image_rgb)
    
    # Detect watermark
    has_watermark, ocr_details = ocr_detector.detect_watermark(image_bgr, threshold=watermark_threshold)
    
    # Detect deformed hands
    if watermark_only:
        has_deformed = False
        hand_details = []
    else:
        has_deformed, hand_details = hand_detector.detect_bad_hands(image_rgb, threshold=hand_threshold)
        
    # Detect multi-headed composition if enabled
    if two_heads:
        has_multi, count, details = dragon_detector.detect_dragon_heads(image_rgb, threshold=dragon_threshold)
    else:
        has_multi = False
        details = []
        
    if has_multi:
        status = "MULTI_HEAD"
        details_str = f"MULTI_HEAD ({count} heads: {', '.join(details)})"
        if has_watermark:
            details_str += f" + WATERMARKED ({', '.join(ocr_details)})"
        if has_deformed:
            details_str += f" + DEFORMED ({', '.join(hand_details)})"
    elif has_watermark and has_deformed:
        status = "COMBINED"
        details_str = f"WATERMARKED & DEFORMED ({', '.join(ocr_details + hand_details)})"
    elif has_watermark:
        status = "WATERMARKED"
        details_str = f"WATERMARKED ({', '.join(ocr_details)})"
    elif has_deformed:
        status = "DEFORMED"
        details_str = f"DEFORMED ({', '.join(hand_details)})"
    else:
        status = "CLEAN"
        details_str = "CLEAN"
        
    return path, status, details_str, filename

def run_sorting():
    watermark_only = "--watermark-only" in sys.argv
    two_heads_only = "--two-heads-only" in sys.argv
    two_heads = "--two-heads" in sys.argv or two_heads_only
    
    print("==================================================================")
    print("              STARTING IMAGE FILTER SORTING RUN                   ")
    if two_heads_only:
        print("               (DRAGON MULTI-HEAD ONLY MODE)                      ")
    elif watermark_only:
        print("                   (WATERMARK ONLY MODE)                          ")
    else:
        print("                  (FULL STANDARD FILTER MODE)                     ")
    print("==================================================================")
    
    # 1. Ensure target directories exist
    dirs_to_create = [config.CORRUPTED_DIR]
    if two_heads_only:
        dirs_to_create.append(config.MULTI_HEAD_DIR)
    elif watermark_only:
        dirs_to_create.append(config.WATERMARKED_DIR)
        if two_heads:
            dirs_to_create.append(config.MULTI_HEAD_DIR)
    else:
        dirs_to_create.extend([config.WATERMARKED_DIR, config.DEFORMED_DIR, config.COMBINED_DIR])
        if two_heads:
            dirs_to_create.append(config.MULTI_HEAD_DIR)
            
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
        
    # 3. Determine number of workers
    # Cap to configured value (NUM_WORKERS) or system count to honor CPU limitations
    num_workers = min(config.NUM_WORKERS, os.cpu_count() or 1)
    print(f"Initializing parallel process pool with {num_workers} worker(s)...")
    
    processed_count = 0
    watermarked_count = 0
    deformed_count = 0
    combined_count = 0
    multi_head_count = 0
    clean_count = 0
    corrupt_count = 0
    
    print("\nProcessing images...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(watermark_only, two_heads_only, two_heads)
    ) as executor:
        
        # Submit all images for processing
        futures = {
            executor.submit(
                process_image_worker,
                path,
                watermark_only,
                two_heads_only,
                two_heads,
                config.WATERMARK_THRESHOLD,
                config.HAND_DEFORMITY_THRESHOLD,
                config.DRAGON_HEAD_THRESHOLD
            ): path for path in image_paths
        }
        
        # Collect and route results sequentially as they complete
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            path, status, details_str, filename = future.result()
            
            # Print immediate result
            print(f"[{processed_count}/{len(image_paths)}] Processing {filename}... -> {details_str}")
            
            # Routing logic (performed on main process to prevent file conflicts)
            if status == "CORRUPT":
                try:
                    utils.move_file(path, config.CORRUPTED_DIR)
                except Exception as e:
                    print(f"    Failed to move corrupt image: {e}")
                corrupt_count += 1
            elif status == "MULTI_HEAD":
                utils.move_file(path, config.MULTI_HEAD_DIR)
                multi_head_count += 1
            elif status == "COMBINED":
                utils.move_file(path, config.COMBINED_DIR)
                combined_count += 1
            elif status == "WATERMARKED":
                utils.move_file(path, config.WATERMARKED_DIR)
                watermarked_count += 1
            elif status == "DEFORMED":
                utils.move_file(path, config.DEFORMED_DIR)
                deformed_count += 1
            else:
                clean_count += 1
                
    print("\n==================================================================")
    print("                       SORTING RUN COMPLETED                      ")
    print("==================================================================")
    print(f"Total processed: {processed_count}")
    print(f"  - Multi-headed (moved to multi_headed/): {multi_head_count}")
    print(f"  - Watermarked only: {watermarked_count}")
    if not watermark_only and not two_heads_only:
        print(f"  - Deformed hands only: {deformed_count}")
        print(f"  - Watermarked & Deformed: {combined_count}")
    print(f"  - Corrupt (moved to corrupted/): {corrupt_count}")
    print(f"  - Clean (remained in source): {clean_count}")
    print("==================================================================")

if __name__ == "__main__":
    run_sorting()
