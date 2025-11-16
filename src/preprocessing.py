import cv2
import numpy as np

IMG_SIZE = 224

def circular_crop(image):
    """
    Applies a circular crop to the image, focusing on the retina.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    cnt = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(cnt)
    
    center_x, center_y = x + w // 2, y + h // 2
    radius = min(w, h) // 2
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    return cropped_image

def prepare_image_for_aug(image_path):
    """
    Loads, resizes, and crops an image.
    Returns a uint8 image, ready for augmentation.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return None
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = circular_crop(image)
    
    return image # Returns uint8 image

def preprocess_for_inference(image_path):
    """
    Full preprocessing pipeline for a single image for prediction.
    This includes the final normalization step.
    """
    # Use the first function to load and prepare the image
    image = prepare_image_for_aug(image_path)
    if image is None:
        return None
    
    # Normalize pixel values for the model
    image = image.astype('float32') / 255.0
    
    return image
