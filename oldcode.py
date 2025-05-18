
# import streamlit as st
# import cv2
# import numpy as np
# from main import segment_and_visualize  # Bone cancer detection
# from fracture_detection import load_model, detect_fractures  # Bone fracture detection

# # Load the YOLOv11 model for bone fracture detection
# try:
#     fracture_model = load_model("best_content4.pt")  # Ensure 'best.pt' is in the same directory
# except Exception as e:
#     st.error(f"Error loading fracture detection model: {e}")
#     fracture_model = None

# # Streamlit App
# st.title("Bone Cancer and Fracture Detection")
# st.write("""
# Upload an X-ray image of a bone, and the models will detect cancerous regions and fractures.
# """)

# # Section 1: Bone Cancer Detection
# st.header("Bone Cancer Detection")
# uploaded_cancer_image = st.file_uploader("Upload an Image for Cancer Detection", type=["jpg", "jpeg", "png", "bmp"])

# if uploaded_cancer_image is not None:
#     # Save the uploaded file temporarily
#     cancer_file_path = f"temp_cancer_image.{uploaded_cancer_image.type.split('/')[-1]}"
#     with open(cancer_file_path, "wb") as f:
#         f.write(uploaded_cancer_image.getbuffer())
    
#     # Display the uploaded image
#     st.image(cancer_file_path, caption="Uploaded Image for Cancer Detection", use_column_width=True)
    
#     # Perform segmentation and visualization
#     if st.button("Detect Cancer"):
#         st.write("Processing cancer detection...")
#         try:
#             # Call the segmentation function
#             segmented_image = segment_and_visualize(cancer_file_path)
            
#             # Display the segmented image
#             st.image(segmented_image, caption="Segmented Image with Cancer Regions", use_column_width=True)
#             st.success("Cancer detection completed successfully!")
#         except Exception as e:
#             st.error(f"An error occurred during cancer detection: {e}")

# # Section 2: Bone Fracture Detection
# st.header("Bone Fracture Detection")
# uploaded_fracture_image = st.file_uploader("Upload an Image for Fracture Detection", type=["jpg", "jpeg", "png", "bmp"])

# if uploaded_fracture_image is not None:
#     # Save the uploaded file temporarily
#     fracture_file_path = f"temp_fracture_image.{uploaded_fracture_image.type.split('/')[-1]}"
#     with open(fracture_file_path, "wb") as f:
#         f.write(uploaded_fracture_image.getbuffer())
    
#     # Display the uploaded image
#     st.image(fracture_file_path, caption="Uploaded Image for Fracture Detection", use_column_width=True)
    
#     # Perform fracture detection
#     if st.button("Detect Fracture"):
#         st.write("Processing fracture detection...")
#         try:
#             if fracture_model is None:
#                 st.error("Fracture detection model is not loaded.")
#             else:
#                 # Call the fracture detection function
#                 fractured_image = detect_fractures(fracture_model, fracture_file_path)
                
#                 # Display the annotated image
#                 st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_column_width=True)
#                 st.success("Fracture detection completed successfully!")
#         except Exception as e:
#             st.error(f"An error occurred during fracture detection: {e}")

# import streamlit as st
# import cv2
# import numpy as np
# from main import segment_and_visualize  # Tumor detection function
# from fracture_detection import load_model, detect_fractures  # Bone fracture detection functions
# import sqlite3
# import torch

# # Database setup
# def init_db():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE,
#             password TEXT,
#             role TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # Add user to database
# def add_user(username, password, role):
#     try:
#         conn = sqlite3.connect('users.db')
#         c = conn.cursor()
#         c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
#         conn.commit()
#         conn.close()
#         return True
#     except sqlite3.IntegrityError:
#         return False

# # Authenticate user
# def authenticate_user(username, password):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
#     user = c.fetchone()
#     conn.close()
#     if user:
#         return {'id': user[0], 'username': user[1], 'role': user[3]}
#     return None

# # Initialize database
# init_db()

# # Load the YOLOv11 model for bone fracture detection
# try:
#     fracture_model = load_model("best_content4.pt")  # Ensure 'best.pt' is in the same directory
# except Exception as e:
#     st.error(f"Error loading fracture detection model: {e}")
#     fracture_model = None

# # Streamlit App
# st.title("Bone Tumor and Fracture Detection")

# # Check if user is logged in
# if 'user' not in st.session_state or st.session_state.user is None:
#     st.subheader("Login or Signup")
#     login_tab, signup_tab = st.tabs(["Login", "Signup"])
    
#     with login_tab:
#         st.header("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             user = authenticate_user(username, password)
#             if user:
#                 st.session_state.user = user
#                 st.success(f"Welcome, {user['username']} ({user['role']})!")
#                 st.rerun()
#             else:
#                 st.error("Invalid username or password.")
    
#     with signup_tab:
#         st.header("Signup")
#         new_username = st.text_input("New Username")
#         new_password = st.text_input("New Password", type="password")
#         role = st.selectbox("Role", ["Doctor", "Patient"])
#         if st.button("Signup"):
#             if add_user(new_username, new_password, role):
#                 st.success("Account created successfully! Please log in.")
#             else:
#                 st.error("Username already exists. Please choose a different username.")
# else:
#     user = st.session_state.user
#     st.write(f"Logged in as {user['username']} ({user['role']})")
#     if st.button("Logout"):
#         st.session_state.user = None
#         st.rerun()

#     st.write("""
#     Upload an X-ray image of a bone, and the models will detect tumor regions and fractures.
#     """)

#     # Section 1: Bone Tumor Detection
#     st.header("Bone Tumor Detection")
#     uploaded_tumor_image = st.file_uploader("Upload an Image for Tumor Detection", type=["jpg", "jpeg", "png", "bmp"])
#     if uploaded_tumor_image is not None:
#         tumor_file_path = f"temp_tumor_image.{uploaded_tumor_image.type.split('/')[-1]}"
#         with open(tumor_file_path, "wb") as f:
#             f.write(uploaded_tumor_image.getbuffer())

#         st.image(tumor_file_path, caption="Uploaded Image for Tumor Detection", use_container_width=True)

#         if st.button("Detect Tumor"):
#             st.write("Processing tumor detection...")
#             try:
#                 segmented_image, predictions = segment_and_visualize(tumor_file_path)
                
#                 if predictions:
#                     st.markdown('<p style="color:skyblue;">Detection Successful: Tumor detected</p>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<p style="color:white;">No detection found: No tumor detected</p>', unsafe_allow_html=True)
                
#                 st.image(segmented_image, caption="Segmented Image with Tumor Regions", use_container_width=True)
#                 st.success("Tumor detection completed successfully!")
#             except Exception as e:
#                 st.error(f"An error occurred during tumor detection: {e}")

#     # Section 2: Bone Fracture Detection
#     st.header("Bone Fracture Detection")
#     uploaded_fracture_image = st.file_uploader("Upload an Image for Fracture Detection", type=["jpg", "jpeg", "png", "bmp"])
#     if uploaded_fracture_image is not None:
#         fracture_file_path = f"temp_fracture_image.{uploaded_fracture_image.type.split('/')[-1]}"
#         with open(fracture_file_path, "wb") as f:
#             f.write(uploaded_fracture_image.getbuffer())

#         st.image(fracture_file_path, caption="Uploaded Image for Fracture Detection", use_container_width=True)

#         if st.button("Detect Fracture"):
#             st.write("Processing fracture detection...")
#             try:
#                 if fracture_model is None:
#                     st.error("Fracture detection model is not loaded.")
#                 else:
#                     # Run inference using YOLO model
#                     results = fracture_model(fracture_file_path)
                    
#                     # Check if any fractures were detected
#                     # Modern YOLO versions store results in a different format
#                     has_detections = False
                    
#                     # Try different methods to access results depending on YOLO version
#                     try:
#                         # Newer YOLO versions
#                         if len(results) > 0:
#                             result = results[0]  # Get the first result (first image)
#                             if hasattr(result, 'boxes') and len(result.boxes) > 0:
#                                 has_detections = len(result.boxes) > 0
#                             elif hasattr(results, 'xyxy') and len(results.xyxy) > 0:
#                                 has_detections = len(results.xyxy[0]) > 0
#                             elif hasattr(result, 'pred') and len(result.pred) > 0:
#                                 has_detections = len(result.pred[0]) > 0
#                     except (AttributeError, IndexError):
#                         # If the above methods fail, try a simpler approach
#                         has_detections = len(results) > 0 and any(len(r.boxes) > 0 for r in results if hasattr(r, 'boxes'))
                    
#                     # Convert results to image
#                     if hasattr(results, 'render'):
#                         # Newer YOLO versions
#                         fractured_image = np.squeeze(results.render())
#                     else:
#                         # Alternative approach
#                         try:
#                             # Try to get the plotted result
#                             fractured_image = results[0].plot()
#                         except:
#                             # If that fails, just use the original image with a warning
#                             fractured_image = cv2.imread(fracture_file_path)
#                             fractured_image = cv2.cvtColor(fractured_image, cv2.COLOR_BGR2RGB)
#                             st.warning("Unable to render detection results. Showing original image.")
                    
#                     # Display results
#                     if has_detections:
#                         st.markdown('<p style="color:skyblue;">Detection Successful: Fracture detected</p>', unsafe_allow_html=True)
#                     else:
#                         st.markdown('<p style="color:white;">No detection found: No fracture detected</p>', unsafe_allow_html=True)

#                     st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_container_width=True)
#                     st.success("Fracture detection completed successfully!")
#             except Exception as e:
#                 st.error(f"An error occurred during fracture detection: {e}")
#                 import traceback
#                 st.error(traceback.format_exc())  # Show detailed error for debugging

import streamlit as st
import cv2
import numpy as np
from main import segment_and_visualize  # Tumor detection function
from fracture_detection import load_model, detect_fractures  # Bone fracture detection functions
import sqlite3
import torch

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject custom CSS
local_css("styles.css")

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Add user to database
def add_user(username, password, role):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return {'id': user[0], 'username': user[1], 'role': user[3]}
    return None

# Initialize database
init_db()

# Load the YOLOv11 model for bone fracture detection
try:
    fracture_model = load_model("best_content4.pt")  # Ensure 'best.pt' is in the same directory
except Exception as e:
    st.error(f"Error loading fracture detection model: {e}")
    fracture_model = None

# Streamlit App
st.title("Bone Tumor and Fracture Detection")

# Check if user is logged in
if 'user' not in st.session_state or st.session_state.user is None:
    st.subheader("Login or Signup")
    login_tab, signup_tab = st.tabs(["Login", "Signup"])
    
    with login_tab:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", key="login_button"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome, {user['username']} ({user['role']})!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    
    with signup_tab:
        st.header("Signup")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["Doctor", "Patient"])
        if st.button("Signup", key="signup_button"):
            if add_user(new_username, new_password, role):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Please choose a different username.")
else:
    user = st.session_state.user
    st.write(f"Logged in as {user['username']} ({user['role']})")
    if st.button("Logout", key="logout_button"):
        st.session_state.user = None
        st.rerun()
    st.write("""
    Upload an X-ray image of a bone, and the models will detect tumor regions and fractures.
    """)
    # Section 1: Bone Tumor Detection
    st.header("Bone Tumor Detection")
    uploaded_tumor_image = st.file_uploader("Upload an Image for Tumor Detection", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_tumor_image is not None:
        tumor_file_path = f"temp_tumor_image.{uploaded_tumor_image.type.split('/')[-1]}"
        with open(tumor_file_path, "wb") as f:
            f.write(uploaded_tumor_image.getbuffer())
        st.image(tumor_file_path, caption="Uploaded Image for Tumor Detection", use_container_width=True)
        if st.button("Detect Tumor", key="detect_tumor_button"):
            st.write("Processing tumor detection...")
            try:
                segmented_image, predictions = segment_and_visualize(tumor_file_path)
                
                if predictions:
                    st.markdown('<p class="success">Detection Successful: Tumor detected</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="info">No detection found: No tumor detected</p>', unsafe_allow_html=True)
                
                st.image(segmented_image, caption="Segmented Image with Tumor Regions", use_container_width=True)
                st.success("Tumor detection completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during tumor detection: {e}")
    # Section 2: Bone Fracture Detection
    st.header("Bone Fracture Detection")
    uploaded_fracture_image = st.file_uploader("Upload an Image for Fracture Detection", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_fracture_image is not None:
        fracture_file_path = f"temp_fracture_image.{uploaded_fracture_image.type.split('/')[-1]}"
        with open(fracture_file_path, "wb") as f:
            f.write(uploaded_fracture_image.getbuffer())
        st.image(fracture_file_path, caption="Uploaded Image for Fracture Detection", use_container_width=True)
        if st.button("Detect Fracture", key="detect_fracture_button"):
            st.write("Processing fracture detection...")
            try:
                if fracture_model is None:
                    st.error("Fracture detection model is not loaded.")
                else:
                    # Run inference using YOLO model
                    results = fracture_model(fracture_file_path)
                    
                    # Check if any fractures were detected
                    has_detections = False
                    
                    try:
                        if len(results) > 0:
                            result = results[0]
                            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                                has_detections = len(result.boxes) > 0
                    except (AttributeError, IndexError):
                        has_detections = False
                    
                    # Convert results to image
                    fractured_image = np.squeeze(results.render()) if hasattr(results, 'render') else cv2.imread(fracture_file_path)
                    fractured_image = cv2.cvtColor(fractured_image, cv2.COLOR_BGR2RGB)
                    
                    # Display results
                    if has_detections:
                        st.markdown('<p class="success">Detection Successful: Fracture detected</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="info">No detection found: No fracture detected</p>', unsafe_allow_html=True)
                    st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_container_width=True)
                    st.success("Fracture detection completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during fracture detection: {e}")
