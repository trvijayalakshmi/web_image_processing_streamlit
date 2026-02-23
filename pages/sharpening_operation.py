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

st.title("Image Edge Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image)
    
    # Convert to grayscale
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Choose structuring element (mask) type
    st.subheader("Structuring Element Configuration")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        mask_type = st.selectbox(
            "Choose Mask Type",
            ["Rectangle (8-connectivity)", "Cross (4-connectivity)", "Ellipse", "Custom"]
        )
    
    with col_b:
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=3, step=2)
    
    # Create structuring element based on selection
    if mask_type == "Rectangle (8-connectivity)":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        connectivity_info = "8-connectivity (all 8 neighboring pixels)"
    elif mask_type == "Cross (4-connectivity)":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        connectivity_info = "4-connectivity (up, down, left, right pixels)"
    elif mask_type == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        connectivity_info = "Elliptical neighborhood"
    else:  # Custom
        # Create a custom diamond-shaped kernel for demonstration
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if abs(i - center) + abs(j - center) <= center:
                    kernel[i, j] = 1
        connectivity_info = "Diamond-shaped (Manhattan distance)"
    
    # Display kernel visualization
    st.write(f"**Connectivity:** {connectivity_info}")
    st.write(f"**Kernel Size:** {kernel_size}x{kernel_size}")
    
    # Show kernel as image
    kernel_display = (kernel * 255).astype(np.uint8)
    kernel_col1, kernel_col2 = st.columns([1, 3])
    with kernel_col1:
        st.write("**Structuring Element:**")
        st.image(kernel_display, width=100)
    
    with kernel_col2:
        st.write("**Kernel Values:**")
        st.code(kernel.tolist())
    
    # Morphological edge detection using gradient
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    eroded = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel)
    edges = cv2.absdiff(dilated, eroded)
    
    with col2:
        st.header("Edge Detection")
        st.image(edges, channels="GRAY")
    
    # Information about morphological edge detection
    st.markdown("---")
    st.subheader("About Morphological Edge Detection")
    
    st.write("""
    **Morphological Gradient Edge Detection:**
    - Uses morphological operations (dilation and erosion) to detect edges
    - Computes the difference between dilated and eroded images
    - Highlights regions of high contrast (edges)
    """)
    
    st.subheader("Structuring Element Types")
    
    mask_info = {
        "Rectangle (8-connectivity)": """
        - **Shape:** Square/rectangular neighborhood
        - **Connectivity:** 8 neighboring pixels (all directions)
        - **Effect:** More inclusive, detects edges in all directions
        - **Use case:** General purpose edge detection
        """,
        
        "Cross (4-connectivity)": """
        - **Shape:** Plus-shaped (+) neighborhood
        - **Connectivity:** 4 neighboring pixels (up, down, left, right)
        - **Effect:** More selective, focuses on horizontal/vertical edges
        - **Use case:** When you want to emphasize cardinal directions
        """,
        
        "Ellipse": """
        - **Shape:** Elliptical neighborhood
        - **Connectivity:** Circular pattern of neighboring pixels
        - **Effect:** Smooth, isotropic edge detection
        - **Use case:** Natural-looking edge detection
        """,
        
        "Custom": """
        - **Shape:** Diamond-shaped (Manhattan distance)
        - **Connectivity:** Based on city-block distance from center
        - **Effect:** Balanced between cross and rectangle
        - **Use case:** Alternative to standard shapes
        """
    }
    
    if mask_type in mask_info:
        st.write(mask_info[mask_type])
    
    st.write(f"""
    **Current Settings:**
    - Mask Type: {mask_type}
    - Kernel Size: {kernel_size}x{kernel_size}
    - Connectivity: {connectivity_info}
    """)