import easyocr

class BorderOCRDetector:
    def __init__(self):
        print("Initializing EasyOCR reader on CPU...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect_watermark(self, image_bgr, threshold=0.40):
        """
        Crop the image into border strips and scan them for text.
        Bypasses central scene text entirely.
        """
        h, w, c = image_bgr.shape
        
        # Calculate border margins
        top_h = int(0.10 * h)
        bottom_h = int(0.10 * h)
        left_w = int(0.08 * w)
        right_w = int(0.08 * w)
        
        # Crop borders
        top_crop = image_bgr[0:top_h, :]
        bottom_crop = image_bgr[h-bottom_h:h, :]
        left_crop = image_bgr[:, 0:left_w]
        right_crop = image_bgr[:, w-right_w:w]
        
        crops = {
            "top": top_crop,
            "bottom": bottom_crop,
            "left": left_crop,
            "right": right_crop
        }
        
        has_watermark = False
        details = []
        
        # Scan only the outer crops
        for position, crop in crops.items():
            if crop.size == 0:
                continue
            # canvas_size=800 for optimal CPU inference speed
            results = self.reader.readtext(crop, canvas_size=800)
            for (bbox, text, prob) in results:
                if prob >= threshold:
                    has_watermark = True
                    details.append(f'[{position}] "{text}" ({prob:.2f})')
                    
        return has_watermark, details
