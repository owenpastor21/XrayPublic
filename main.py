# app.py

# Import necessary libraries
import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import io

# =====================================
# Configuration Variables
# =====================================

# Path to the trained YOLOv8 Large classification model
trained_model_path = 'best.pt'  # Updated path

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Automatically use GPU if available

# =====================================
# Function Definitions
# =====================================

@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLO model from the specified path.

    Parameters:
    - model_path (str): Path to the trained YOLO model weights.

    Returns:
    - model: Loaded YOLO model or None if loading fails.
    """
    if not torch.cuda.is_available():
        st.warning("CUDA device not found. Using CPU for inference. This may be slower.")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_image(model, image):
    """
    Performs inference on the uploaded image.

    Parameters:
    - model: The trained YOLO model.
    - image (PIL.Image): Uploaded image.

    Returns:
    - predicted_class (str): Predicted class name.
    - confidence (float): Confidence level of the prediction.
    """
    try:
        # Perform prediction directly on the PIL image
        results = model.predict(image, verbose=False)

        if not results:
            return "No Prediction", 0.0

        # Extract predicted class and confidence using Probs attributes
        probs = results[0].probs  # Access the Probs object

        # Check if probs has the necessary attributes
        if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
            predicted_class = results[0].names[probs.top1]
            confidence = probs.top1conf * 100  # Convert to percentage
            return predicted_class, confidence
        else:
            st.error("Probs object does not have 'top1' or 'top1conf' attributes.")
            return "Error", 0.0

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error", 0.0

# =====================================
# Streamlit App Layout
# =====================================

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="Chest X-ray Image Classifier by BGPastor",
        page_icon="🖼️",
        layout="centered",
        initial_sidebar_state="collapsed",  # Collapse sidebar if it exists
    )

    st.title("🖼️ Chest X-ray Image Classifier by BGPastor")
    st.write("""
    Upload an image, and the model will predict its class along with the confidence level.
    """)

    # =====================================
    # Display Sample X-ray Images
    # =====================================

    st.markdown("### 📷 **Sample X-ray Images**")
    st.write("Here are some examples of the types of X-ray images you can upload:")

    # List of sample image paths
    sample_image_paths = [
        "ss_(1).jpeg",  # Sample Image 1
        "ss_(2).jpeg",  # Sample Image 2
        "ss_(3).jpeg",  # Sample Image 3
        "ss_(4).jpeg"   # Sample Image 4
    ]

    # Create a single row with four columns for the sample images
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            try:
                img = Image.open(sample_image_paths[i]).convert("RGB")
                st.image(img, width=150, caption=f"Sample {i+1}")
            except Exception as e:
                st.error(f"Error loading sample image {i+1}: {e}")

    st.markdown("---")  # Horizontal line separator

    # =====================================
    # Load the model
    # =====================================
    model = load_model(trained_model_path)
    if model is None:
        st.stop()  # Stop execution if model failed to load

    # =====================================
    # Image Input: Upload or Capture
    # =====================================

    st.markdown("### 📤 **Upload or Capture an Image**")

    # Create two columns for upload and camera input
    input_cols = st.columns([1, 1])

    with input_cols[0]:
        # File uploader for image
        uploaded_file = st.file_uploader(
            "📁 **Upload an image...**",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            accept_multiple_files=False
        )

    with input_cols[1]:
        # Camera input for capturing image
        captured_image = st.camera_input("📸 **Capture an image using your camera...**")

    # Determine which input to use: uploaded_file or captured_image
    if uploaded_file is not None:
        input_image = uploaded_file
    elif captured_image is not None:
        input_image = captured_image
    else:
        input_image = None

    if input_image is not None:
        try:
            # Open the image
            image = Image.open(input_image).convert("RGB")

            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Perform prediction
            with st.spinner('🔍 Performing inference...'):
                predicted_class, confidence = predict_image(model, image)

            # Display the results
            if predicted_class != "Error":
                st.success(f"**Predicted Class:** {predicted_class}")
                st.info(f"**Confidence Level:** {confidence:.2f}%")
            else:
                st.error("Prediction failed due to an internal error.")

        except Exception as e:
            st.error(f"⚠️ Error processing the image: {e}")

    else:
        st.info("ℹ️ Please upload or capture an image to get started.")

if __name__ == '__main__':
    main()
