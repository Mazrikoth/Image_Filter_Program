import os
import glob
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def test_owlvit():
    print("Loading OWL-ViT model...")
    # Load OWL-ViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.eval()
    
    # Get a few test images from the source folder containing "dragon"
    source_dir = config.SOURCE_DIR
    image_paths = glob.glob(os.path.join(source_dir, "*dragon*"))[:50]
    
    if not image_paths:
        print("No dragon images found to test. Falling back to first 10 images.")
        image_paths = glob.glob(os.path.join(source_dir, "*.png"))[:10]
        
    if not image_paths:
        print("No images found to test.")
        return
        
    texts = [["a dragon head"]]
    
    print("\nRunning inference...")
    for path in image_paths:
        filename = os.path.basename(path)
        print(f"\nImage: {filename}")
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(text=texts, images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Target image sizes (height, width) to rescale box coordinates
            target_sizes = torch.Tensor([image.size[::-1]])
            # Retrieve predictions with objectness score and functional threshold
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.10
            )
            
            i = 0  # Retrieve predictions for the first image
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            
            detected_heads = 0
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                # Label 0 corresponds to "a dragon head" (our text query)
                if score >= 0.15:
                    detected_heads += 1
                    print(f"  Detected dragon head with score: {score:.3f} at box {box}")
            
            print(f"Total detected dragon heads: {detected_heads}")
            
        except Exception as e:
            print(f"  Error processing image: {e}")

if __name__ == "__main__":
    test_owlvit()
