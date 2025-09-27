import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import os

# ==============================
# Load YOLOv8 model
# ==============================
# Update this to your local path
MODEL_PATH = r"C:\Users\Gouthum\Downloads\glovee\best.pt"

# Initialize model
model = YOLO(MODEL_PATH)

# ==============================
# Streamlit UI
# ==============================
st.title("🧤 Glove Detection App")
st.write("Upload an image or video to detect gloves using YOLOv8.")

# Sidebar options
option = st.sidebar.radio("Choose Input Type:", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Run YOLO detection
        results = model.predict(img)

        # Save annotated image
        annotated_img = results[0].plot()  
        st.image(annotated_img, caption="Detected Gloves", use_column_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()  # placeholder for video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model.predict(frame)

            # Draw detections
            annotated_frame = results[0].plot()

            # Display frame in Streamlit
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
