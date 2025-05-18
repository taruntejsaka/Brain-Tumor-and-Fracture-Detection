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

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS for navigation bar
nav_css = """
<style>
    .navbar {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .nav-item {
        padding: 8px 16px;
        border-radius: 5px;
        text-decoration: none;
        color: #0066cc;
        font-weight: bold;
    }
    .nav-item:hover {
        background-color: #e6f0ff;
    }
    .nav-item.active {
        background-color: #0066cc;
        color: white;
    }
    .user-info {
        margin-top: 5px;
        padding: 5px 10px;
        text-align: right;
        font-size: 0.9em;
        color: #666;
    }
</style>
"""

# Database functions
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    
    # Create reports table if it doesn't exist
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
    print("Database initialized")

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
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Debug print
        print(f"Saving report: {patient_id}, {doctor_id}, {report_date}, {diagnosis_type}, {diagnosis_result}, {image_path}, {notes}")
        
        c.execute('''
            INSERT INTO reports (patient_id, doctor_id, report_date, diagnosis_type, diagnosis_result, image_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, doctor_id, report_date, diagnosis_type, diagnosis_result, image_path, notes))
        
        conn.commit()
        report_id = c.lastrowid
        conn.close()
        print(f"Report saved with ID: {report_id}")
        return True
    except Exception as e:
        print(f"Error saving report: {e}")
        return False

# Get reports for a specific patient or doctor
def get_reports(user_id, role):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        if role == "Doctor":
            query = '''
                SELECT r.id, p.username, r.report_date, r.diagnosis_type, r.diagnosis_result, r.image_path, r.notes
                FROM reports r
                JOIN users p ON r.patient_id = p.id
                WHERE r.doctor_id = ?
                ORDER BY r.report_date DESC
            '''
        else:  # Patient
            query = '''
                SELECT r.id, d.username, r.report_date, r.diagnosis_type, r.diagnosis_result, r.image_path, r.notes
                FROM reports r
                JOIN users d ON r.doctor_id = d.id
                WHERE r.patient_id = ?
                ORDER BY r.report_date DESC
            '''
        
        c.execute(query, (user_id,))
        reports_data = c.fetchall()
        conn.close()
        
        # Format the reports as a list of dictionaries for easier access
        reports = []
        for report in reports_data:
            reports.append({
                'id': report[0],
                'related_user': report[1],  # patient_name or doctor_name
                'report_date': report[2],
                'diagnosis_type': report[3],
                'diagnosis_result': report[4],
                'image_path': report[5],
                'notes': report[6]
            })
        
        print(f"Retrieved {len(reports)} reports for user {user_id}")
        return reports
    except Exception as e:
        print(f"Error getting reports: {e}")
        return []

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
    if image_path and os.path.exists(image_path):
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

# Initialize the database when the app starts
init_db()

# Load the YOLOv11 model for bone fracture detection
try:
    fracture_model = load_model("best_content4.pt")  # Ensure 'best.pt' is in the same directory
except Exception as e:
    st.error(f"Error loading fracture detection model: {e}")
    fracture_model = None

# Initialize session state variables if they don't exist
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "diagnosis_image_path" not in st.session_state:
    st.session_state.diagnosis_image_path = None
if "diagnosis_type" not in st.session_state:
    st.session_state.diagnosis_type = None
if "report_saved" not in st.session_state:
    st.session_state.report_saved = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Diagnosis"

# Streamlit App
st.title("Bone Tumor and Fracture Detection System")

# Check if user is logged in
if 'user' not in st.session_state or st.session_state.user is None:
    st.subheader("Login or Signup")
    login_tab, signup_tab = st.tabs(["Login", "Signup"])
    
    with login_tab:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
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
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["Doctor", "Patient"], key="signup_role")
        if st.button("Signup", key="signup_button"):
            if add_user(new_username, new_password, role):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Please choose a different username.")
else:
    user = st.session_state.user
    
    # Navigation bar
    st.markdown(nav_css, unsafe_allow_html=True)
    
    if user['role'] == "Doctor":
        navigation_items = ["Diagnosis", "Reports", "Export Data"]
    else:
        navigation_items = ["Diagnosis", "Export Data"]
    
    # Create the navigation bar HTML
    nav_html = '<div class="navbar">'
    for nav_item in navigation_items:
        active_class = "active" if st.session_state.current_page == nav_item else ""
        nav_html += f'<a href="#" class="nav-item {active_class}" onclick="this.preventDefault(); window.parent.postMessage({{command: \'streamlit:setComponentValue\', key: \'nav_choice\', value: \'{nav_item}\'}}, \'*\');">{nav_item}</a>'
    nav_html += '</div>'
    
    # Add user info and logout button
    nav_html += f'<div class="user-info">Logged in as {user["username"]} ({user["role"]}) &nbsp; '
    nav_html += f'<a href="#" onclick="window.parent.postMessage({{command: \'streamlit:setComponentValue\', key: \'logout_clicked\', value: true}}, \'*\');">Logout</a></div>'
    
    # Render navigation
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Handle navigation clicks
    nav_choice = st.text_input("", key="nav_choice", label_visibility="collapsed")
    if nav_choice and nav_choice != st.session_state.current_page:
        st.session_state.current_page = nav_choice
        st.rerun()
    
    # Handle logout
    logout_clicked = st.text_input("", key="logout_clicked", label_visibility="collapsed")
    if logout_clicked == "true":
        st.session_state.user = None
        st.rerun()
    
    # Display the current page content
    if st.session_state.current_page == "Diagnosis":
        st.write("""
        Upload an X-ray image of a bone, and the models will detect tumor regions and fractures.
        """)
        
        # Create tabs for different diagnosis types
        diagnosis_tab = st.radio("Select Diagnosis Type", ["Bone Tumor Detection", "Bone Fracture Detection"])
        
        # Reset report_saved state when changing diagnosis type
        if "previous_diagnosis_tab" not in st.session_state:
            st.session_state.previous_diagnosis_tab = diagnosis_tab
        elif st.session_state.previous_diagnosis_tab != diagnosis_tab:
            st.session_state.report_saved = False
            st.session_state.previous_diagnosis_tab = diagnosis_tab
        
        if diagnosis_tab == "Bone Tumor Detection":
            st.header("Bone Tumor Detection")
            uploaded_tumor_image = st.file_uploader("Upload an Image for Tumor Detection", type=["jpg", "jpeg", "png", "bmp"], key="tumor_image")
            
            if uploaded_tumor_image is not None:
                tumor_file_path = f"temp_tumor_image.{uploaded_tumor_image.type.split('/')[-1]}"
                with open(tumor_file_path, "wb") as f:
                    f.write(uploaded_tumor_image.getbuffer())
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(tumor_file_path, caption="Uploaded Image for Tumor Detection", use_column_width=True)
                
                with col2:
                    if st.button("Detect Tumor", key="detect_tumor_button", use_container_width=True):
                        # Reset report_saved flag when new detection is performed
                        st.session_state.report_saved = False
                        
                        st.write("Processing tumor detection...")
                        try:
                            segmented_image, predictions = segment_and_visualize(tumor_file_path)
                            
                            # Save diagnostic result in session state
                            st.session_state.diagnosis_type = "Bone Tumor Detection"
                            st.session_state.diagnosis_result = "Tumor detected" if predictions else "No tumor detected"
                            
                            if predictions:
                                st.success("Detection Successful: Tumor detected")
                            else:
                                st.info("No tumor detected in the image")
                            
                            # Save the segmented image
                            segmented_image_path = f"segmented_tumor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            plt.imsave(segmented_image_path, segmented_image)
                            st.session_state.diagnosis_image_path = segmented_image_path
                            
                            st.image(segmented_image, caption="Segmented Image with Tumor Regions", use_column_width=True)
                            st.success("Tumor detection completed successfully!")
                            
                        except Exception as e:
                            st.error(f"An error occurred during tumor detection: {e}")
                
                # Only show the report saving section if detection has been performed
                if st.session_state.diagnosis_result and user['role'] == "Doctor" and not st.session_state.report_saved:
                    st.subheader("Save Report for Patient")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        patients = get_all_patients()
                        if not patients:
                            st.warning("No patients registered in the system. Please ask patients to create accounts.")
                        else:
                            patient_options = {p[1]: p[0] for p in patients}
                            selected_patient = st.selectbox("Select Patient", list(patient_options.keys()), key="tumor_patient_select")
                            
                    with col2:
                        notes = st.text_area("Additional Notes", key="tumor_notes")
                    
                    save_button = st.button("Save Patient Report", key="save_tumor_report_btn", use_container_width=True)
                    if save_button:
                        patient_id = patient_options[selected_patient]
                        doctor_id = user['id']
                        diagnosis_type = st.session_state.diagnosis_type
                        diagnosis_result = st.session_state.diagnosis_result
                        image_path = st.session_state.diagnosis_image_path
                        
                        if save_report(
                            patient_id,
                            doctor_id,
                            diagnosis_type,
                            diagnosis_result,
                            image_path,
                            notes
                        ):
                            st.session_state.report_saved = True
                            st.success(f"Report saved successfully for patient {selected_patient}!")
                        else:
                            st.error("Failed to save the report. Please try again.")
        
        elif diagnosis_tab == "Bone Fracture Detection":
            st.header("Bone Fracture Detection")
            uploaded_fracture_image = st.file_uploader("Upload an Image for Fracture Detection", type=["jpg", "jpeg", "png", "bmp"], key="fracture_image")
            
            if uploaded_fracture_image is not None:
                fracture_file_path = f"temp_fracture_image.{uploaded_fracture_image.type.split('/')[-1]}"
                with open(fracture_file_path, "wb") as f:
                    f.write(uploaded_fracture_image.getbuffer())
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(fracture_file_path, caption="Uploaded Image for Fracture Detection", use_column_width=True)
                
                with col2:
                    if st.button("Detect Fracture", key="detect_fracture_button", use_container_width=True):
                        # Reset report_saved flag when new detection is performed
                        st.session_state.report_saved = False
                        
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
                                    st.success("Detection Successful: Fracture detected")
                                else:
                                    st.info("No fracture detected in the image")
                                
                                st.image(fractured_image, caption="Annotated Image with Fracture Regions", use_column_width=True)
                                st.success("Fracture detection completed successfully!")
                                
                        except Exception as e:
                            st.error(f"An error occurred during fracture detection: {e}")
                
                # Only show the report saving section if detection has been performed
                if st.session_state.diagnosis_result and user['role'] == "Doctor" and not st.session_state.report_saved:
                    st.subheader("Save Report for Patient")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        patients = get_all_patients()
                        if not patients:
                            st.warning("No patients registered in the system. Please ask patients to create accounts.")
                        else:
                            patient_options = {p[1]: p[0] for p in patients}
                            selected_patient = st.selectbox("Select Patient", list(patient_options.keys()), key="fracture_patient_select")
                    
                    with col2:
                        notes = st.text_area("Additional Notes", key="fracture_notes")
                    
                    save_button = st.button("Save Patient Report", key="save_fracture_report_btn", use_container_width=True)
                    if save_button:
                        patient_id = patient_options[selected_patient]
                        doctor_id = user['id']
                        diagnosis_type = st.session_state.diagnosis_type
                        diagnosis_result = st.session_state.diagnosis_result
                        image_path = st.session_state.diagnosis_image_path
                        
                        if save_report(
                            patient_id,
                            doctor_id,
                            diagnosis_type,
                            diagnosis_result,
                            image_path,
                            notes
                        ):
                            st.session_state.report_saved = True
                            st.success(f"Report saved successfully for patient {selected_patient}!")
                        else:
                            st.error("Failed to save the report. Please try again.")
    
    elif st.session_state.current_page == "Reports" and user['role'] == "Doctor":
        st.header("Patient Reports")
        
        # Force refresh reports data when accessing the page
        if st.button("Refresh Reports", key="refresh_reports"):
            st.rerun()
        
        # Display reports for doctors
        reports = get_reports(user['id'], user['role'])
        
        if not reports:
            st.info("No reports found. Diagnose patients to create reports.")
        else:
            # Group reports by patient for better organization
            patient_reports = {}
            for report in reports:
                patient_name = report['related_user']
                if patient_name not in patient_reports:
                    patient_reports[patient_name] = []
                patient_reports[patient_name].append(report)
            
            # Create tabs for each patient
            for patient_name, patient_reports_list in patient_reports.items():
                with st.expander(f"Reports for {patient_name}"):
                    for i, report in enumerate(patient_reports_list):
                        st.write("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Date:** {report['report_date']}")
                            st.write(f"**Diagnosis Type:** {report['diagnosis_type']}")
                            st.write(f"**Result:** {report['diagnosis_result']}")
                            
                            if report['notes']:
                                st.write(f"**Notes:** {report['notes']}")
                            
                            # Generate PDF report button
                            if st.button(f"Generate PDF Report", key=f"pdf_{report['id']}_{i}", use_container_width=True):
                                pdf_path = create_pdf_report(
                                    patient_name,
                                    user['username'],  # doctor name
                                    report['diagnosis_type'],
                                    report['diagnosis_result'],
                                    report['image_path'],
                                    report['notes']
                                )
                                
                                st.markdown(get_download_link(pdf_path, "Download PDF Report", "pdf"), unsafe_allow_html=True)
                        
                        with col2:
                            if report['image_path'] and os.path.exists(report['image_path']):
                                st.image(report['image_path'], caption=f"Diagnostic Image", use_column_width=True)
                            else:
                                st.warning(f"Image not found: {report['image_path']}")
    
    elif st.session_state.current_page == "Export Data":
        st.header("Export Data")
        
        # Get all reports for the current user
        reports = get_reports(user['id'], user['role'])
        
        if not reports:
            st.info("No data available to export.")
        else:
            # Export options
            export_format = st.radio("Select Export Format", ["CSV", "PDF", "JSON"])
            
            if export_format == "CSV":
                if st.button("Generate CSV Export", use_container_width=True):
                    csv_filename = f"{user['role'].lower()}_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    # Convert reports to DataFrame
                    df = pd.DataFrame([
                        {
                            'Report ID': r['id'],
                            'Patient/Doctor': r['related_user'],
                            'Date': r['report_date'],
                            'Diagnosis Type': r['diagnosis_type'],
                            'Result': r['diagnosis_result'],
                            'Notes': r['notes']
                        } for r in reports
                    ])
                    df.to_csv(csv_filename, index=False)
                    
                    st.success(f"CSV file generated: {csv_filename}")
                    st.markdown(get_download_link(csv_filename, "Download CSV File", "csv"), unsafe_allow_html=True)
            
            elif export_format == "PDF":
                st.write("This will generate a comprehensive PDF report containing all your reports.")
                if st.button("Generate PDF Report", use_container_width=True):
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
                        all_reports_pdf.set_font('Arial', 'B', 14)
                        all_reports_pdf.cell(0, 10, f"{report['diagnosis_type']} - {report['report_date']}", 0, 1, 'L')
                        
                        all_reports_pdf.set_font('Arial', 'B', 12)
                        all_reports_pdf.cell(50, 10, 'Patient/Doctor:', 0)
                        all_reports_pdf.set_font('Arial', '', 12)
                        all_reports_pdf.cell(0, 10, report['related_user'], 0, 1)
                        
                        all_reports_pdf.set_font('Arial', 'B', 12)
                        all_reports_pdf.cell(50, 10, 'Result:', 0)
                        all_reports_pdf.set_font('Arial', '', 12)
                        all_reports_pdf.cell(0, 10, report['diagnosis_result'], 0, 1)
                        
                        if report['notes']:
                            all_reports_pdf.set_font('Arial', 'B', 12)
                            all_reports_pdf.cell(50, 10, 'Notes:', 0)
                            all_reports_pdf.set_font('Arial', '', 12)
                            all_reports