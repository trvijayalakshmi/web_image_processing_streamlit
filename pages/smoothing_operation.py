import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .stTitle {
        color: #2e86c1;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stHeader {
        color: #1b4f72;
        font-family: 'Arial', sans-serif;
    }
    .stImage {
        border: 2px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-1d391kg {  /* File uploader styling */
        background-color: #ecf0f1;
        border: 1px solid #bdc3c7;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Image Smoothing Operations")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(image)

    # Convert to OpenCV format
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Smoothing operation selection
    smoothing_type = st.selectbox(
        "Choose Smoothing Operation",
        ["Mean Blur", "Gaussian Blur", "Median Blur", "Bilateral Filter"]
    )

    # Parameters based on selected operation
    if smoothing_type == "Mean Blur":
        st.subheader("Mean Blur Parameters")
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=5, step=2)
        if st.button("Apply Mean Blur"):
            # Apply mean blur
            smoothed = cv2.blur(img_array, (kernel_size, kernel_size))
            smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)
            with col2:
                st.header("Mean Blurred Image")
                st.image(smoothed_rgb)

    elif smoothing_type == "Gaussian Blur":
        st.subheader("Gaussian Blur Parameters")
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=5, step=2)
        sigma = st.slider("Sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        if st.button("Apply Gaussian Blur"):
            # Apply Gaussian blur
            smoothed = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)
            with col2:
                st.header("Gaussian Blurred Image")
                st.image(smoothed_rgb)

    elif smoothing_type == "Median Blur":
        st.subheader("Median Blur Parameters")
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=5, step=2)
        if st.button("Apply Median Blur"):
            # Apply median blur
            smoothed = cv2.medianBlur(img_array, kernel_size)
            smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)
            with col2:
                st.header("Median Blurred Image")
                st.image(smoothed_rgb)

    elif smoothing_type == "Bilateral Filter":
        st.subheader("Bilateral Filter Parameters")
        diameter = st.slider("Diameter", min_value=5, max_value=20, value=9)
        sigma_color = st.slider("Sigma Color", min_value=10, max_value=150, value=75)
        sigma_space = st.slider("Sigma Space", min_value=10, max_value=150, value=75)
        if st.button("Apply Bilateral Filter"):
            # Apply bilateral filter
            smoothed = cv2.bilateralFilter(img_array, diameter, sigma_color, sigma_space)
            smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)
            with col2:
                st.header("Bilateral Filtered Image")
                st.image(smoothed_rgb)

    # Information about the selected operation
    st.markdown("---")
    st.subheader("About the Selected Operation")

    if smoothing_type == "Mean Blur":
        st.write("""
        **Mean Blur** (Average Filtering):
        - Replaces each pixel value with the average of its neighboring pixels
        - Simple and fast smoothing operation
        - Can reduce noise but may blur edges
        - Kernel size determines the extent of smoothing
        """)

    elif smoothing_type == "Gaussian Blur":
        st.write("""
        **Gaussian Blur**:
        - Uses a Gaussian kernel for weighted averaging
        - Weights decrease with distance from center pixel
        - Better at preserving edges compared to mean blur
        - Sigma controls the spread of the Gaussian function
        """)

    elif smoothing_type == "Median Blur":
        st.write("""
        **Median Blur**:
        - Replaces each pixel with the median value of neighboring pixels
        - Excellent for removing salt-and-pepper noise
        - Preserves edges better than linear filters
        - Particularly effective for impulse noise
        """)

    elif smoothing_type == "Bilateral Filter":
        st.write("""
        **Bilateral Filter**:
        - Advanced smoothing that preserves edges
        - Considers both spatial proximity and intensity similarity
        - Reduces noise while maintaining sharp edges
        - More computationally intensive but produces better results
        """)
