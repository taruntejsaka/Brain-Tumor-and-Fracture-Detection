import streamlit as st
import cv2
import numpy as np
from main import segment_and_visualize  # Tumor detection function
from fracture_detection import load_model, detect_fractures  # Bone fracture detection functions
import sqlite3
import torch
import pandas as pd
import base64
from datetime import datetime
import io
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import json

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Database functions (existing code)
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
    
    # Create a new table for patient reports
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_id INTEGER,
            report_date TEXT,
            diagnosis_type TEXT,
            diagnosis_result TEXT,
            image_path TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# Add user to database (existing code)
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

# Authenticate user (existing code)
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return {'id': user[0], 'username': user[1], 'role': user[3]}
    return None

# Get all patients (for doctors to select from)
def get_all_patients():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username FROM users WHERE role = "Patient"')
    patients = c.fetchall()
    conn.close()
    return patients

# Save report to database
def save_report(patient_id, doctor_id, diagnosis_type, diagnosis_result, image_path, notes):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO reports (patient_id, doctor_id, report_date, diagnosis_type, diagnosis_result, image_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, doctor_id, report_date, diagnosis_type, diagnosis_result, image_path, notes))
    conn.commit()
    conn.close()
    return True

# Get reports for a specific patient or doctor
def get_reports(user_id, role):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    if role == "Doctor":
        c.execute('''
            SELECT r.id, u.username, r.report_date, r.diagnosis_type, r.diagnosis_result, r.image_path, r.notes
            FROM reports r
            JOIN users u ON r.patient_id = u.id
            WHERE r.doctor_id = ?
            ORDER BY r.report_date DESC
        ''', (user_id,))
    else:  # Patient
        c.execute('''
            SELECT r.id, u.username, r.report_date, r.diagnosis_type, r.diagnosis_result, r.image_path, r.notes
            FROM reports r
            JOIN users u ON r.doctor_id = u.id
            WHERE r.patient_id = ?
            ORDER BY r.report_date DESC
        ''', (user_id,))
    
    reports = c.fetchall()
    conn.close()
    return reports

# Function to create a downloadable link
def get_download_link(file_path, label, file_format="csv"):
    with open(file_path, "rb") as file:
        contents = file.read()
    
    b64 = base64.b64encode(contents).decode()
    filename = os.path.basename(file_path)
    mime_type = "text/csv" if file_format == "csv" else "application/pdf"
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{label}</a>'
    return href

# Function to generate PDF report
def create_pdf_report(patient_name, doctor_name, diagnosis_type, diagnosis_result, image_path, notes):
    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Medical Diagnostic Report', 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Report details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Patient Name:', 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, patient_name, 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Doctor Name:', 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, doctor_name, 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Date:', 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Diagnosis Type:', 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, diagnosis_type, 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Diagnosis Result:', 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, diagnosis_result, 0, 1)
    
    # Add image if it exists
    if os.path.exists(image_path):
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Diagnostic Image:', 0, 1)
        pdf.image(image_path, x=10, y=None, w=180)
    
    # Add notes
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Notes:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, notes)
    
    # Save the PDF to a file
    report_filename = f"report_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_filename)
    return report_filename

# Function to export reports to CSV
def export_reports_to_csv(reports, filename):
    # Convert reports to DataFrame
    df = pd.DataFrame(reports, columns=['Report ID', 'Patient/Doctor', 'Date', 'Diagnosis Type', 'Result', 'Image Path', 'Notes'])
    # Save to CSV
    df.to_csv(filename, index=False)
    return filename

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
    st.sidebar.write(f"Logged in as {user['username']} ({user['role']})")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.user = None
        st.rerun()
    
    # Navigation menu in sidebar
    page = st.sidebar.radio("Navigation", ["Diagnosis", "Reports", "Export Data"])
    
    if page == "Diagnosis":
        st.write("""
        Upload an X-ray image of a bone, and the models will detect tumor regions and fractures.
        """)
        
        # Create tabs for different diagnosis types
        diagnosis_tab = st.radio("Select Diagnosis Type", ["Bone Tumor Detection", "Bone Fracture Detection"])
        
        # Initialize session variables for saving diagnosis data
        if "diagnosis_result" not in st.session_state:
            st.session_state.diagnosis_result = None
        if "diagnosis_image_path" not in st.session_state:
            st.session_state.diagnosis_image_path = None
        if "diagnosis_type" not in st.session_state:
            st.session_state.diagnosis_type = None
        
        if diagnosis_tab == "Bone Tumor Detection":
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
                        
                        # Save diagnostic result in session state
                        st.session_state.diagnosis_type = "Bone Tumor Detection"
                        st.session_state.diagnosis_result = "Tumor detected" if predictions else "No tumor detected"
                        st.session_state.diagnosis_image_path = tumor_file_path
                        
                        if predictions:
                            st.markdown('<p class="success">Detection Successful: Tumor detected</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="info">No detection found: No tumor detected</p>', unsafe_allow_html=True)
                        
                        # Save the segmented image
                        segmented_image_path = f"segmented_tumor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        plt.imsave(segmented_image_path, segmented_image)
                        st.session_state.diagnosis_image_path = segmented_image_path
                        
                        st.image(segmented_image, caption="Segmented Image with Tumor Regions", use_container_width=True)
                        st.success("Tumor detection completed successfully!")
                        
                        # Option to save report
                        if user['role'] == "Doctor":
                            st.subheader("Save Report")
                            patients = get_all_patients()
                            patient_options = {p[1]: p[0] for p in patients}
                            selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
                            notes = st.text_area("Additional Notes")
                            
                            if st.button("Save Report"):
                                save_report(
                                    patient_options[selected_patient],
                                    user['id'],
                                    st.session_state.diagnosis_type,
                                    st.session_state.diagnosis_result,
                                    st.session_state.diagnosis_image_path,
                                    notes
                                )
                                st.success("Report saved successfully!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during tumor detection: {e}")
        
        elif diagnosis_tab == "Bone Fracture Detection":
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
                            
                            # Process the result properly
                            if len(results) > 0:
                                result = results[0]  # Get the first result
                                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                                    has_detections = True
                                    
                                    # Get original image for drawing custom boxes
                                    original_img = cv2.imread(fracture_file_path)
                                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                    
                                    # Draw green boxes manually
                                    for box in result.boxes.xyxy:  # Get bounding boxes in xyxy format
                                        x1, y1, x2, y2 = map(int, box.tolist())
                                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                                        cv2.putText(original_img, "Fracture", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    
                                    fractured_image = original_img
                                else:
                                    # No detections, use original image
                                    fractured_image = cv2.imread(fracture_file_path)
                                    fractured_image = cv2.cvtColor(fractured_image, cv2.COLOR_BGR2RGB)
                            else:
                                # No results, use original image
                                fractured_image = cv2.imread(fracture_file_path)
                                fractured_image = cv2.cvtColor(fractured_image, cv2.COLOR_BGR2RGB)
                            
                            # Save diagnostic result in session state
                            st.session_state.diagnosis_type = "Bone Fracture Detection"
                            st.session_state.diagnosis_result = "Fracture detected" if has_detections else "No fracture detected"
                            
                            # Save the processed image
                            fractured_image_path = f"fractured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(fractured_image_path, cv2.cvtColor(fractured_image, cv2.COLOR_RGB2BGR))
                            st.session_state.diagnosis_image_path = fractured_image_path
                            
                            # Display results
                            if has_detections:
                                st.markdown('<p class="success">Detection Successful: Fracture detected</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="info">No detection found: No fracture detected</p>', unsafe_allow_html=True)
                            
                            st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_container_width=True)
                            st.success("Fracture detection completed successfully!")
                            
                            # Option to save report
                            if user['role'] == "Doctor":
                                st.subheader("Save Report")
                                patients = get_all_patients()
                                patient_options = {p[1]: p[0] for p in patients}
                                selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
                                notes = st.text_area("Additional Notes")
                                
                                if st.button("Save Report"):
                                    save_report(
                                        patient_options[selected_patient],
                                        user['id'],
                                        st.session_state.diagnosis_type,
                                        st.session_state.diagnosis_result,
                                        st.session_state.diagnosis_image_path,
                                        notes
                                    )
                                    st.success("Report saved successfully!")
                    
                    except Exception as e:
                        st.error(f"An error occurred during fracture detection: {e}")
    
    elif page == "Reports":
        st.header("Medical Reports")
        
        # Display reports based on user role
        reports = get_reports(user['id'], user['role'])
        
        if not reports:
            st.info("No reports found.")
        else:
            for report in reports:
                report_id, related_user, report_date, diagnosis_type, diagnosis_result, image_path, notes = report
                
                with st.expander(f"{diagnosis_type} - {report_date}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if user['role'] == "Doctor":
                            st.write(f"**Patient:** {related_user}")
                        else:
                            st.write(f"**Doctor:** {related_user}")
                        st.write(f"**Date:** {report_date}")
                        st.write(f"**Diagnosis Type:** {diagnosis_type}")
                        st.write(f"**Result:** {diagnosis_result}")
                        
                        if notes:
                            st.write(f"**Notes:** {notes}")
                    
                    with col2:
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Diagnostic Image", use_column_width=True)
                        else:
                            st.warning("Image not found")
                    
                    # Generate PDF report
                    if st.button(f"Generate PDF Report", key=f"pdf_{report_id}"):
                        patient_name = related_user if user['role'] == "Doctor" else user['username']
                        doctor_name = user['username'] if user['role'] == "Doctor" else related_user
                        
                        pdf_path = create_pdf_report(
                            patient_name,
                            doctor_name,
                            diagnosis_type,
                            diagnosis_result,
                            image_path,
                            notes
                        )
                        
                        st.markdown(get_download_link(pdf_path, "Download PDF Report", "pdf"), unsafe_allow_html=True)
    
    elif page == "Export Data":
        st.header("Export Data")
        
        # Get all reports for the current user
        reports = get_reports(user['id'], user['role'])
        
        if not reports:
            st.info("No data available to export.")
        else:
            # Export options
            export_format = st.radio("Select Export Format", ["CSV", "PDF", "JSON"])
            
            if export_format == "CSV":
                if st.button("Generate CSV Export"):
                    csv_filename = f"{user['role'].lower()}_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    export_path = export_reports_to_csv(reports, csv_filename)
                    st.success(f"CSV file generated: {csv_filename}")
                    st.markdown(get_download_link(export_path, "Download CSV File", "csv"), unsafe_allow_html=True)
            
            elif export_format == "PDF":
                st.write("This will generate a comprehensive PDF report containing all your reports.")
                if st.button("Generate PDF Report"):
                    # Create a PDF with all reports
                    all_reports_pdf = FPDF()
                    all_reports_pdf.add_page()
                    
                    # Title
                    all_reports_pdf.set_font('Arial', 'B', 16)
                    all_reports_pdf.cell(0, 10, f'Medical Reports Summary - {user["username"]}', 0, 1, 'C')
                    all_reports_pdf.line(10, 30, 200, 30)
                    all_reports_pdf.ln(10)
                    
                    # Reports
                    for report in reports:
                        report_id, related_user, report_date, diagnosis_type, diagnosis_result, image_path, notes = report
                        
                        all_reports_pdf.set_font('Arial', 'B', 14)
                        all_reports_pdf.cell(0, 10, f"{diagnosis_type} - {report_date}", 0, 1, 'L')
                        
                        all_reports_pdf.set_font('Arial', 'B', 12)
                        all_reports_pdf.cell(50, 10, 'Patient/Doctor:', 0)
                        all_reports_pdf.set_font('Arial', '', 12)
                        all_reports_pdf.cell(0, 10, related_user, 0, 1)
                        
                        all_reports_pdf.set_font('Arial', 'B', 12)
                        all_reports_pdf.cell(50, 10, 'Result:', 0)
                        all_reports_pdf.set_font('Arial', '', 12)
                        all_reports_pdf.cell(0, 10, diagnosis_result, 0, 1)
                        
                        if notes:
                            all_reports_pdf.set_font('Arial', 'B', 12)
                            all_reports_pdf.cell(50, 10, 'Notes:', 0)
                            all_reports_pdf.set_font('Arial', '', 12)
                            all_reports_pdf.multi_cell(0, 10, notes)
                        
                        if os.path.exists(image_path):
                            all_reports_pdf.image(image_path, x=10, y=None, w=180)
                        
                        all_reports_pdf.ln(10)
                        all_reports_pdf.line(10, all_reports_pdf.get_y(), 200, all_reports_pdf.get_y())
                        all_reports_pdf.ln(10)
                    
                    # Save PDF
                    all_reports_filename = f"all_reports_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    all_reports_pdf.output(all_reports_filename)
                    
                    st.success(f"PDF report generated: {all_reports_filename}")
                    st.markdown(get_download_link(all_reports_filename, "Download PDF Report", "pdf"), unsafe_allow_html=True)
            
            elif export_format == "JSON":
                if st.button("Generate JSON Export"):
                    # Convert reports to JSON format
                    json_data = []
                    for report in reports:
                        report_id, related_user, report_date, diagnosis_type, diagnosis_result, image_path, notes = report
                        json_data.append({
                            "id": report_id,
                            "related_user": related_user,
                            "date": report_date,
                            "diagnosis_type": diagnosis_type,
                            "result": diagnosis_result,
                            "notes": notes
                        })
                    
                    # Save to JSON file
                    json_filename = f"{user['role'].lower()}_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(json_filename, 'w') as json_file:
                        json.dump(json_data, json_file, indent=4)
                    
                    st.success(f"JSON file generated: {json_filename}")
                    
                    # Create download link
                    with open(json_filename, "rb") as file:
                        contents = file.read()
                    
                    b64 = base64.b64encode(contents).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{json_filename}">Download JSON File</a>'
                    st.markdown(href, unsafe_allow_html=True)
