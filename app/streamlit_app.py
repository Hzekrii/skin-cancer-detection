"""
Streamlit web app for interactive skin cancer classification.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.gradcam import GradCAM

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="🔬 Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    model_path = config.MODEL_CHECKPOINT_PATH
    if not os.path.exists(model_path):
        st.error(
            f"❌ Model not found at `{model_path}`\n\n"
            f"Please train the model first:\n"
            f"```\npython main.py --mode train\n```"
        )
        st.stop()
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_gradcam(_model):
    """Initialize Grad-CAM (cached)."""
    return GradCAM(_model)


def preprocess_image(uploaded_file):
    """Preprocess an uploaded image."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(config.IMAGE_SIZE)
    img_array = np.array(img)
    img_display = img_array.copy()
    img_normalized = img_array.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_display


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/microscope.png",
        width=80
    )
    st.title("About")
    st.markdown("""
    **Skin Cancer Classifier** uses deep learning to analyze 
    dermatoscopic images of skin lesions.
    
    ---
    
    **🧠 Model Details:**
    - **Architecture:** MobileNetV2
    - **Method:** Transfer Learning
    - **Dataset:** HAM10000
    - **Classes:** 7 lesion types
    - **Explainability:** Grad-CAM
    
    ---
    
    **📊 Performance:**
    - Accuracy: ~70%
    - Mean AUC: 0.90
    
    ---
    
    **⚠️ Disclaimer:**  
    This tool is for **educational purposes only**.  
    Always consult a qualified dermatologist.
    """)

    st.markdown("---")
    st.markdown("**🔗 Links:**")
    st.markdown("[📂 GitHub Repository](https://github.com/Hzekrii/skin-cancer-detection)")
    st.markdown("[📄 HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)")


# ─── Main Content ────────────────────────────────────────
st.markdown("<h1 class='main-title'>🔬 Skin Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Transfer Learning + Explainable AI (Grad-CAM)</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ─── Upload Section ──────────────────────────────────────
st.subheader("📁 Upload a Dermatoscopic Image")
st.markdown(
    "Upload a `.jpg` or `.png` image of a skin lesion for analysis."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a dermatoscopic image of a skin lesion"
)

# ─── Lesion Info ─────────────────────────────────────────
with st.expander("ℹ️ About the 7 Lesion Categories"):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        | Code | Category | Type |
        |------|----------|------|
        | `nv` | Melanocytic Nevi | Benign |
        | `bkl` | Benign Keratosis | Benign |
        | `df` | Dermatofibroma | Benign |
        | `vasc` | Vascular Lesions | Benign |
        """)
    with col_info2:
        st.markdown("""
        | Code | Category | Type |
        |------|----------|------|
        | `mel` | Melanoma | **Malignant** |
        | `bcc` | Basal Cell Carcinoma | **Malignant** |
        | `akiec` | Actinic Keratoses | Pre-cancerous |
        """)

st.markdown("---")

# ─── Analysis ────────────────────────────────────────────
if uploaded_file is not None:
    # Load model
    model = load_model()
    gradcam_obj = load_gradcam(model)

    # Preprocess
    img_batch, img_display = preprocess_image(uploaded_file)

    # Predict
    with st.spinner("🔄 Analyzing image..."):
        preds = model.predict(img_batch, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]
        pred_label = config.CLASSES[pred_class]
        pred_full = config.CLASS_LABELS[pred_label]

        # Grad-CAM
        heatmap = gradcam_obj.compute_heatmap(img_batch, pred_index=pred_class)
        superimposed, heatmap_colored = gradcam_obj.overlay_heatmap(
            heatmap, img_display
        )

    # ─── Results ─────────────────────────────────────────
    st.subheader("🎯 Analysis Results")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**📷 Original Image**")
        st.image(img_display, use_container_width=True)

    with col2:
        st.markdown("**🔍 Grad-CAM Heatmap**")
        st.image(heatmap_colored, use_container_width=True)

    with col3:
        st.markdown("**🎯 Grad-CAM Overlay**")
        st.image(superimposed, use_container_width=True)

    st.markdown("---")

    # ─── Prediction Details ──────────────────────────────
    col_pred1, col_pred2 = st.columns([1, 2])

    with col_pred1:
        st.subheader("📋 Diagnosis")

        # Color based on severity
        danger_classes = ["mel", "bcc", "akiec"]
        if pred_label in danger_classes:
            st.error(f"⚠️ **{pred_full}**")
            if pred_label == "mel":
                st.warning("Melanoma is a serious form of skin cancer. Please consult a dermatologist immediately.")
            elif pred_label == "bcc":
                st.warning("Basal Cell Carcinoma requires medical attention.")
            elif pred_label == "akiec":
                st.warning("Actinic Keratoses can be pre-cancerous. Please consult a doctor.")
        else:
            st.success(f"✅ **{pred_full}**")
            st.info("This appears to be a benign lesion, but regular monitoring is recommended.")

        st.metric("Confidence", f"{confidence:.1%}")

    with col_pred2:
        st.subheader("📊 All Probabilities")

        # Sort by probability
        sorted_indices = np.argsort(preds[0])[::-1]

        for idx in sorted_indices:
            cls = config.CLASSES[idx]
            prob = preds[0][idx]
            full_name = config.CLASS_LABELS[cls]

            # Color bar based on type
            if cls in danger_classes:
                bar_label = f"⚠️ {full_name}: {prob:.1%}"
            else:
                bar_label = f"✅ {full_name}: {prob:.1%}"

            st.progress(float(prob), text=bar_label)

    # ─── Grad-CAM Explanation ────────────────────────────
    st.markdown("---")
    with st.expander("🧠 How does Grad-CAM work?"):
        st.markdown("""
        **Grad-CAM** (Gradient-weighted Class Activation Mapping) 
        is an explainable AI technique that highlights the regions 
        of the image that were most important for the model's prediction.
        
        **How it works:**
        1. The model makes a prediction
        2. Gradients are computed from the predicted class back to the last convolutional layer
        3. These gradients are averaged to get importance weights for each feature map
        4. A weighted combination of feature maps creates the heatmap
        5. The heatmap is overlaid on the original image
        
        **🔴 Red/Yellow regions** = Most important for the prediction  
        **🔵 Blue regions** = Less important
        
        This helps doctors understand **why** the model made its prediction,
        building trust and enabling verification.
        """)

    # ─── Warning ─────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer:** This tool is for educational and research "
        "purposes only. It is NOT a substitute for professional medical diagnosis. "
        "Always consult a qualified dermatologist for any skin concerns."
    )

else:
    # ─── Demo / Instructions ─────────────────────────────
    st.info("👆 Upload an image above to get started!")

    st.markdown("### 🖼️ What kind of images work best?")
    st.markdown("""
    - **Dermatoscopic images** (taken with a dermatoscope)
    - Close-up photos of skin lesions
    - Well-lit, in-focus images
    - Images similar to the HAM10000 dataset
    """)

    # Show sample results if available
    gradcam_grid_path = os.path.join(config.FIGURES_DIR, "gradcam_grid.png")
    if os.path.exists(gradcam_grid_path):
        st.markdown("### 📊 Sample Grad-CAM Results")
        st.image(gradcam_grid_path, use_container_width=True)

    confusion_matrix_path = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        st.markdown("### 📈 Model Performance")
        st.image(confusion_matrix_path, use_container_width=True)