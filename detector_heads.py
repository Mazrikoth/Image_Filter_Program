import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class DragonHeadDetector:
    def __init__(self):
        print("Initializing OWL-ViT dragon head detector (google/owlvit-base-patch32)...")
        try:
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.model.eval()
        except Exception as e:
            print(f"Error loading OWL-ViT model: {e}")
            raise e

    def detect_dragon_heads(self, image_rgb, threshold=0.15):
        """
        Detect dragon heads using OWL-ViT.
        Returns:
            has_multi_heads (bool): True if 2 or more dragon heads are detected.
            detected_count (int): Number of detected dragon heads.
            details (list): Confidence list for each detection.
        """
        if isinstance(image_rgb, Image.Image):
            pil_image = image_rgb
        else:
            pil_image = Image.fromarray(image_rgb)
            
        texts = [["a dragon head"]]
        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        
        # We call the post-processing helper directly on the image_processor
        results = self.processor.image_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        detected_count = len(scores)
        has_multi_heads = detected_count >= 2
        
        details = []
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            box_coords = [round(coord, 1) for coord in box.tolist()]
            details.append(f"Head_{idx+1}({score*100:.1f}%)")
            
        return has_multi_heads, detected_count, details
