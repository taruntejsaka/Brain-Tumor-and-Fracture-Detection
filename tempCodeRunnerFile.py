import streamlit as st
import cv2
import numpy as np
from main import segment_and_visualize  # Bone cancer detection
from fracture_detection import load_model, detect_fractures  # Bone fracture detection

# Load the YOLOv11 model for bone fracture detection
try:
    fracture_model = load_model("best_content4.pt")  # Ensure 'best.pt' is in the same directory
except Exception as e:
    st.error(f"Error loading fracture detection model: {e}")
    fracture_model = None

# Streamlit App
st.title("Bone Cancer and Fracture Detection")
st.write("""
Upload an X-ray image of a bone, and the models will detect cancerous regions and fractures.
""")

# Section 1: Bone Cancer Detection
st.header("Bone Cancer Detection")
uploaded_cancer_image = st.file_uploader("Upload an Image for Cancer Detection", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_cancer_image is not None:
    # Save the uploaded file temporarily
    cancer_file_path = f"temp_cancer_image.{uploaded_cancer_image.type.split('/')[-1]}"
    with open(cancer_file_path, "wb") as f:
        f.write(uploaded_cancer_image.getbuffer())
    
    # Display the uploaded image
    st.image(cancer_file_path, caption="Uploaded Image for Cancer Detection", use_column_width=True)
    
    # Perform segmentation and visualization
    if st.button("Detect Cancer"):
        st.write("Processing cancer detection...")
        try:
            # Call the segmentation function
            segmented_image = segment_and_visualize(cancer_file_path)
            
            # Display the segmented image
            st.image(segmented_image, caption="Segmented Image with Cancer Regions", use_column_width=True)
            st.success("Cancer detection completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during cancer detection: {e}")

# Section 2: Bone Fracture Detection
st.header("Bone Fracture Detection")
uploaded_fracture_image = st.file_uploader("Upload an Image for Fracture Detection", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_fracture_image is not None:
    # Save the uploaded file temporarily
    fracture_file_path = f"temp_fracture_image.{uploaded_fracture_image.type.split('/')[-1]}"
    with open(fracture_file_path, "wb") as f:
        f.write(uploaded_fracture_image.getbuffer())
    
    # Display the uploaded image
    st.image(fracture_file_path, caption="Uploaded Image for Fracture Detection", use_column_width=True)
    
    # Perform fracture detection
    if st.button("Detect Fracture"):
        st.write("Processing fracture detection...")
        try:
            if fracture_model is None:
                st.error("Fracture detection model is not loaded.")
            else:
                # Call the fracture detection function
                fractured_image = detect_fractures(fracture_model, fracture_file_path)
                
                # Display the annotated image
                st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_column_width=True)
                st.success("Fracture detection completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during fracture detection: {e}")