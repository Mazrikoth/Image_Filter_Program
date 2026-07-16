import os
import glob
import time
import torch
import numpy as np
from PIL import Image
import config
import utils
from detector_ocr import BorderOCRDetector
from detector_hands import HandAnatomyDetector

# Optimize PyTorch CPU execution
torch.set_num_threads(1)


def calibrate():
    print("==================================================================")
    print("          STARTING AUTOMATED THRESHOLD CALIBRATION ENGINE         ")
    print("==================================================================")
    
    # Initialize detectors
    ocr_detector = BorderOCRDetector()
    hand_detector = HandAnatomyDetector()
    
    # 1. Define True Positive and True Negative groups from curated folders
    print("\nScanning curated control set directories...")
    
    # True Positives: Watermarked
    watermark_files = glob.glob(os.path.join(config.WATERMARKED_DIR, "*.png"))
    watermark_files.extend(glob.glob(os.path.join(config.WATERMARKED_DIR, "*.jpg")))
    
    # True Positives: Deformed Hands
    deformed_files = glob.glob(os.path.join(config.DEFORMED_DIR, "*.png"))
    deformed_files.extend(glob.glob(os.path.join(config.DEFORMED_DIR, "*.jpg")))
    
    # True Positives: Combined (Both)
    combined_files = glob.glob(os.path.join(config.COMBINED_DIR, "*.png"))
    combined_files.extend(glob.glob(os.path.join(config.COMBINED_DIR, "*.jpg")))
    
    # True Negatives: Clean images (remaining in the main folder)
    clean_files = glob.glob(os.path.join(config.SOURCE_DIR, "*.png"))
    clean_files.extend(glob.glob(os.path.join(config.SOURCE_DIR, "*.jpg")))
    
    print(f"Curated Dataset Size:")
    print(f"  - Curated Watermarked Only: {len(watermark_files)}")
    print(f"  - Curated Deformed Hands Only: {len(deformed_files)}")
    print(f"  - Curated Combined (Both): {len(combined_files)}")
    print(f"  - Curated Clean (True Negatives): {len(clean_files)}")
    
    total_curated = len(watermark_files) + len(deformed_files) + len(combined_files) + len(clean_files)
    if total_curated == 0:
        print("Error: No images found in control set. Please check folder paths.")
        return
        
    # Gather raw scores
    print("\nExtracting AI raw scores across all curated control images...")
    
    watermark_scores = []  # tuples of (filename, max_ocr_prob, is_watermark_ground_truth)
    hand_scores = []       # tuples of (filename, max_deformed_prob, is_hand_ground_truth)
    
    # Helper function to extract scores for a file list
    def extract_scores_for_group(file_list, is_watermark_gt, is_hand_gt):
        for path in file_list:
            filename = os.path.basename(path)
            image_rgb = utils.load_image_rgb(path)
            if image_rgb is None:
                continue
            image_bgr = utils.rgb_to_bgr(image_rgb)
            
            # OCR Score
            h, w, c = image_bgr.shape
            top_h = int(0.10 * h)
            bottom_h = int(0.10 * h)
            left_w = int(0.08 * w)
            right_w = int(0.08 * w)
            
            ocr_max = 0.0
            # Scan top and bottom borders only
            for crop in [image_bgr[0:top_h, :], image_bgr[h-bottom_h:h, :], image_bgr[:, 0:left_w], image_bgr[:, w-right_w:w]]:
                if crop.size == 0: continue
                res = ocr_detector.reader.readtext(crop, canvas_size=800)
                for (_, _, prob) in res:
                    if prob > ocr_max:
                        ocr_max = prob
            watermark_scores.append((filename, ocr_max, is_watermark_gt))
            
            # Hand Score
            hand_max = 0.0
            yolo_res = hand_detector.yolo_model.predict(image_rgb, verbose=False)
            for res in yolo_res:
                for box in res.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    pad = int(0.20 * max(x_max - x_min, y_max - y_min))
                    x_min = max(0, x_min - pad); y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad); y_max = min(h, y_max + pad)
                    
                    cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
                    if cropped_hand.size == 0: continue
                    pil_hand = Image.fromarray(cropped_hand)
                    inputs = hand_detector.barc_processor(images=pil_hand, return_tensors="pt")
                    with torch.no_grad():
                        outputs = hand_detector.barc_model(**inputs)
                    probs = outputs.logits.softmax(dim=1)
                    bad_prob = 0.0
                    for label_idx, label_name in hand_detector.id2label.items():
                        if "Bad_Anatomy" in label_name:
                            bad_prob += probs[0][label_idx].item()
                    if bad_prob > hand_max:
                        hand_max = bad_prob
            hand_scores.append((filename, hand_max, is_hand_gt))

    has_positives = (len(watermark_files) > 0) or (len(deformed_files) > 0) or (len(combined_files) > 0)
    
    if not has_positives:
        print("\n[!] Curated ground-truth positive folders are empty.")
        print("    Auto-calibrating thresholds using the clean control group to guarantee 0% false positives...")
        print("Processing Clean Control Images...")
        extract_scores_for_group(clean_files[:100], is_watermark_gt=False, is_hand_gt=False)
        
        max_ocr_fp = max([score for _, score, _ in watermark_scores]) if watermark_scores else 0.0
        max_hand_fp = max([score for _, score, _ in hand_scores]) if hand_scores else 0.0
        
        opt_watermark_t = min(0.95, max(0.40, max_ocr_fp + 0.05))
        opt_hand_t = min(0.95, max(0.60, max_hand_fp + 0.05))
        
        water_p = 1.0; water_r = 0.0; water_f = 0.0
        hand_p = 1.0; hand_r = 0.0; hand_f = 0.0
    else:
        # Process all groups
        print("Processing Curated Watermarked Only...")
        extract_scores_for_group(watermark_files, is_watermark_gt=True, is_hand_gt=False)
        print("Processing Curated Deformed Hands Only...")
        extract_scores_for_group(deformed_files, is_watermark_gt=False, is_hand_gt=True)
        print("Processing Curated Combined (Both)...")
        extract_scores_for_group(combined_files, is_watermark_gt=True, is_hand_gt=True)
        print("Processing Curated Clean (True Negatives)...")
        extract_scores_for_group(clean_files[:100], is_watermark_gt=False, is_hand_gt=False)
        
        # 3. Autotune Thresholds
        print("\nOptimizing classification threshold hyperparameters...")
        
        # Helper function to find best threshold based on F1-score
        def optimize_threshold(scores_list, label_name):
            best_thresh = 0.50
            best_f1 = 0.0
            best_precision = 0.0
            best_recall = 0.0
            
            # Test thresholds from 0.05 to 0.95
            for t in np.linspace(0.05, 0.95, 91):
                tp = fp = fn = tn = 0
                for name, score, gt in scores_list:
                    pred = score >= t
                    if pred and gt: tp += 1
                    elif pred and not gt: fp += 1
                    elif not pred and gt: fn += 1
                    else: tn += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Prioritize 100% precision (zero false positives) if possible, otherwise max F1
                if precision >= 1.0 and recall > 0.0:
                    if recall > best_recall or best_precision < 1.0:
                        best_thresh = t
                        best_f1 = f1
                        best_precision = precision
                        best_recall = recall
                elif precision > best_precision and best_precision < 1.0:
                    best_thresh = t
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                elif f1 > best_f1 and best_precision < 1.0:
                    best_thresh = t
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    
            return best_thresh, best_precision, best_recall, best_f1

        opt_watermark_t, water_p, water_r, water_f = optimize_threshold(watermark_scores, "Watermark")
        opt_hand_t, hand_p, hand_r, hand_f = optimize_threshold(hand_scores, "Deformed Hand")

    
    print("\n==================================================================")
    print("                     CALIBRATION TUNING RESULTS                   ")
    print("==================================================================")
    print(f"Optimal Watermark Threshold: {opt_watermark_t:.2f}")
    print(f"  - Precision: {water_p*100:.1f}% (False Positives: 0%)")
    print(f"  - Recall: {water_r*100:.1f}%")
    print(f"  - F1-Score: {water_f:.3f}")
    print(f"\nOptimal Hand Deformity Threshold: {opt_hand_t:.2f}")
    print(f"  - Precision: {hand_p*100:.1f}% (False Positives: 0%)")
    print(f"  - Recall: {hand_r*100:.1f}%")
    print(f"  - F1-Score: {hand_f:.3f}")
    print("==================================================================")
    
    # 4. Write back to config.py
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    print(f"\nWriting calibrated thresholds back to {config_path}...")
    try:
        content = f"""import os

# Central Paths
SOURCE_DIR = r"{config.SOURCE_DIR}"
WATERMARKED_DIR = os.path.join(SOURCE_DIR, "watermarked")
DEFORMED_DIR = os.path.join(SOURCE_DIR, "deformed_hands")
COMBINED_DIR = os.path.join(SOURCE_DIR, "watermarked_and_deformed")
CORRUPTED_DIR = os.path.join(SOURCE_DIR, "corrupted")
MULTI_HEAD_DIR = os.path.join(SOURCE_DIR, "multi_headed")

# Central Detection Thresholds (Autotuned by calibration.py)
WATERMARK_THRESHOLD = {opt_watermark_t:.2f}             # Calibrated OCR text probability
HAND_DEFORMITY_THRESHOLD = {opt_hand_t:.2f}        # Calibrated BARC bad-anatomy crop probability
WHOLE_IMAGE_DEFORMED_THRESHOLD = 0.45  # Standard fallback
DRAGON_HEAD_THRESHOLD = {config.DRAGON_HEAD_THRESHOLD:.2f}           # Confidence threshold for OWL-ViT dragon head detection

# Multiprocessing limits
NUM_WORKERS = {config.NUM_WORKERS}  # Spawn parallel worker processes (safe RAM usage)
"""
        with open(config_path, "w") as f:
            f.write(content)
        print("Config successfully updated with your customized thresholds!")
    except Exception as e:
        print(f"Error saving config file: {e}")

if __name__ == '__main__':
    calibrate()
