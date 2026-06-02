import streamlit as st
import numpy as np
import os

from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load Models
# --------------------------------------------------

enhanced_model = load_model(
    "muffin_chihuahua_enhanced.keras"
)

transfer_model = load_model(
    "muffin_chihuahua_transfer.keras"
)

# --------------------------------------------------
# Demo Images
# --------------------------------------------------

sample_images = {
    "🧁 Muffin Example 1": "demopic1.png",
    "🐶 Chihuahua Example 1": "demopic2.png",
    "🧁 Muffin Example 2": "demopic3.png",
    "🐶 Chihuahua Example 2": "demopic4.png",
    "🐶 Chihuahua Example 3": "demopic5.png",
    "🧁 Muffin Example 3": "demopic6.png"
}

# --------------------------------------------------
# Image Settings
# --------------------------------------------------

IMG_SIZE = (128, 128)

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def prepare_image(image):

    image = image.resize(IMG_SIZE)

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🐶🧁 Muffin vs Chihuahua")

st.write(
    "Compare predictions from the Enhanced CNN and "
    "Transfer Learning CNN models."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Sample Images")

    selected_sample = st.selectbox(
        "Choose a demo image:",
        ["None"] + list(sample_images.keys())
    )

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

st.caption(
    "Try the built-in demo images from the sidebar or upload your own image."
)

# --------------------------------------------------
# Determine Image Source
# --------------------------------------------------

image = None

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

elif selected_sample != "None":

    image_path = os.path.join(
        "demo_images",
        sample_images[selected_sample]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

# --------------------------------------------------
# Predict
# --------------------------------------------------

if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    image_array = prepare_image(image)

    enhanced_prob = enhanced_model.predict(
        image_array,
        verbose=0
    )[0][0]

    transfer_prob = transfer_model.predict(
        image_array,
        verbose=0
    )[0][0]

    enhanced_label = (
        "🧁 Muffin"
        if enhanced_prob > 0.5
        else "🐶 Chihuahua"
    )

    transfer_label = (
        "🧁 Muffin"
        if transfer_prob > 0.5
        else "🐶 Chihuahua"
    )

    enhanced_confidence = (
        max(enhanced_prob, 1 - enhanced_prob) * 100
    )

    transfer_confidence = (
        max(transfer_prob, 1 - transfer_prob) * 100
    )

    # --------------------------------------------------
    # Side-by-side Comparison
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Enhanced CNN")

        st.metric(
            "Prediction",
            enhanced_label
        )

        st.metric(
            "Confidence",
            f"{enhanced_confidence:.1f}%"
        )

    with col2:

        st.subheader("Transfer Learning CNN")

        st.metric(
            "Prediction",
            transfer_label
        )

        st.metric(
            "Confidence",
            f"{transfer_confidence:.1f}%"
        )

    st.divider()

    st.subheader("Model Performance")

    st.write("Enhanced CNN Accuracy: **92.57%**")
    st.write("Transfer Learning CNN Accuracy: **98.06%**")