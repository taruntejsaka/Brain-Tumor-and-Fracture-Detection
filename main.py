# from roboflow import Roboflow
# import cv2
# import tkinter as tk
# from tkinter import filedialog

# # Initialize Roboflow with API key
# rf = Roboflow(api_key="XMzSc3R9Tuw3GC2qCX2J")
# project = rf.workspace().project("bone-cancer-segmentation")
# model = project.version(1).model

# def segment_and_visualize(image_path, output_path):
#     """
#     Segments the bone cancer region, visualizes it on the image, and saves the output.
#     """
#     # Get the prediction from the model
#     prediction = model.predict(image_path, confidence=35)
    
#     # Read the original image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Iterate through each prediction and draw it
#     for pred in prediction.json()['predictions']:
#         # Extract details
#         x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
#         confidence = pred['confidence']
#         label = pred['class']
        
#         # Draw a rectangle or segmentation mask
#         cv2.rectangle(
#             image,
#             (int(x - width / 2), int(y - height / 2)),
#             (int(x + width / 2), int(y + height / 2)),
#             (0, 255, 0),
#             2
#         )
        
#         # Add confidence and label
#         cv2.putText(
#             image,
#             f"{label} {confidence * 100:.1f}%",
#             (int(x - width / 2), int(y - height / 2) - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 0),
#             2
#         )
    
#     # Save the resulting image
#     output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_path, output_image)
#     print(f"Segmented image saved at {output_path}")

# # Function to open file dialog and get the input image path
# def select_image():
#     root = tk.Tk()
#     root.withdraw()  # Hide the main tkinter window
#     file_path = filedialog.askopenfilename(
#         title="Select an Image",
#         filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
#     )
#     return file_path

# # Main execution
# if __name__ == "__main__":
#     # Let the user select an image
#     input_image = select_image()
    
#     if not input_image:
#         print("No image selected. Exiting...")
#     else:
#         # Define the output path
#         output_image = "segmented_output.jpg"
        
#         # Perform segmentation and visualization
#         segment_and_visualize(input_image, output_image)


from roboflow import Roboflow
import cv2
import numpy as np

# Initialize Roboflow with API key
rf = Roboflow(api_key="XMzSc3R9Tuw3GC2qCX2J")
project = rf.workspace().project("bone-cancer-segmentation")
model = project.version(1).model

def segment_and_visualize(image_path):
    """
    Segments the bone cancer region, visualizes it on the image, and returns the processed image.
    """
    # Get the prediction from the model
    prediction = model.predict(image_path, confidence=40)
    
    # Read the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    return image