import cv2
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification

class HandAnatomyDetector:
    def __init__(self):
        print("Initializing YOLOv8 hand bounding box detector...")
        # Download the community standard hand_yolov8n.pt directly from Hugging Face hub
        try:
            model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="hand_yolov8n.pt")
            self.yolo_model = YOLO(model_path)
        except Exception as e:
            print(f"Error downloading hand_yolov8n.pt: {e}")
            raise e

        print("Initializing BARC Vision Transformer (angusleung100/bad-anatomy-realism-classifier)...")
        try:
            self.barc_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.barc_model = AutoModelForImageClassification.from_pretrained("angusleung100/bad-anatomy-realism-classifier")
            self.id2label = self.barc_model.config.id2label
        except Exception as e:
            print(f"Error loading BARC model: {e}")
            raise e

    def detect_bad_hands(self, image_rgb, threshold=0.50):
        """
        Detect hands using YOLOv8, crop them, and grade anatomy using the BARC ViT.
        Completely eliminates false positives from backgrounds.
        """
        # Run YOLOv8 inference
        results = self.yolo_model.predict(image_rgb, verbose=False)
        
        has_deformed_hand = False
        details = []
        h, w, c = image_rgb.shape
        
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, xyxy)
                
                # Add 20% padding to hand crop
                pad = int(0.20 * max(x_max - x_min, y_max - y_min))
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                
                cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
                if cropped_hand.size == 0:
                    continue
                    
                # Classify hand crop
                pil_hand = Image.fromarray(cropped_hand)
                inputs = self.barc_processor(images=pil_hand, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.barc_model(**inputs)
                
                probs = outputs.logits.softmax(dim=1)
                
                # Aggregate probabilities for all 'Bad Anatomy' classes
                bad_prob = 0.0
                class_details = []
                for label_idx, label_name in self.id2label.items():
                    prob = probs[0][label_idx].item()
                    class_details.append(f"{label_name}: {prob*100:.1f}%")
                    if "Bad Anatomy" in label_name:
                        bad_prob += prob
                
                if bad_prob >= threshold:
                    has_deformed_hand = True
                    details.append(f"Hand_{idx+1}_deformed({bad_prob*100:.1f}%)")
                else:
                    details.append(f"Hand_{idx+1}_clean({(1.0 - bad_prob)*100:.1f}%)")
                    
        return has_deformed_hand, details
