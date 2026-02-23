import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Morphological Operations", layout="wide")

st.title("Morphological Operations")

# Sidebar for operation selection
st.sidebar.header("Settings")
operation = st.sidebar.selectbox(
    "Select Morphological Operation",
    ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
)

kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, step=2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply selected operation
    if operation == "Erosion":
        result = cv2.erode(gray, kernel)
    elif operation == "Dilation":
        result = cv2.dilate(gray, kernel)
    elif operation == "Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif operation == "Gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif operation == "Top Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif operation == "Black Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader(f"{operation} Result")
        st.image(result, use_column_width=True, clamp=True)
else:
    st.info("Please upload an image to get started")