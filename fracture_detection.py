from ultralytics import YOLO
import cv2
import os

def load_model(model_path):
    """
    Loads the YOLOv11 model from the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return YOLO(model_path)

def detect_fractures(model, image_path, confidence_threshold=0.30):
    """
    Detects bone fractures in the given image and returns the annotated image.
    
    Args:
        model: The loaded YOLOv11 model.
        image_path (str): Path to the input image.
        confidence_threshold (float): Minimum confidence score for detections (default: 0.35).
    
    Returns:
        annotated_image: Image with detected fractures annotated.
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path '{image_path}'.")
    
    # Run inference on the image with the specified confidence threshold
    results = model(image, conf=confidence_threshold)
    
    # Annotate the image with the results
    annotated_image = results[0].plot()
    return annotated_image

