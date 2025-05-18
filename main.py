
from roboflow import Roboflow
import cv2
import numpy as np

# Initialize Roboflow with API key
rf = Roboflow(api_key="XMzSc3R9Tuw3GC2qCX2J")
project = rf.workspace().project("bone-cancer-segmentation")
model = project.version(1).model

def segment_and_visualize(image_path):
    """
    Segments the bone cancer region, visualizes it on the image, and returns the processed image and predictions.
    """
    # Get the prediction from the model
    prediction = model.predict(image_path, confidence=40)
    
    # Read the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictions = []
    
    # Iterate through each prediction and draw it
    for pred in prediction.json()['predictions']:
        # Extract details
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        confidence = pred['confidence']
        label = pred['class']
        
        # Draw a rectangle or segmentation mask
        cv2.rectangle(
            image,
            (int(x - width / 2), int(y - height / 2)),
            (int(x + width / 2), int(y + height / 2)),
            (0, 255, 0),  # Green rectangle
            2
        )
        
        # Add confidence and label
        cv2.putText(
            image,
            f"{label} {confidence * 100:.1f}%",
            (int(x - width / 2), int(y - height / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),  # Yellow text
            2
        )
        
        # Append prediction details to the list
        predictions.append({
            'label': label,
            'confidence': confidence,
            'bbox': (x, y, width, height)
        })
    
    return image, predictions  # Return both the image and the predictions