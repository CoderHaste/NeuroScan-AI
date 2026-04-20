import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import matplotlib.pyplot as plt
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import os
import gdown

MODEL_PATH = "brain_tumor_model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=19mGV7eKw8oMSQxwtypWQtbyfHv_iNJFQ"
    gdown.download(url, MODEL_PATH, quiet=False)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="NeuroScan AI",
    layout="centered",
    page_icon="🧠"
)

# CUSTOM STYLING (UI)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 0px 30px rgba(56,189,248,0.2);
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

img {
    border-radius: 12px;
    border: 2px solid rgba(56,189,248,0.3);
}

.good { color: #22c55e; font-size: 26px; text-align:center; }
.bad { color: #ef4444; font-size: 26px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">NeuroScan AI</div>', unsafe_allow_html=True)

# LABELS & DESCRIPTIONS
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

descriptions = {
    "glioma": "Aggressive tumor originating in glial cells.",
    "meningioma": "Typically benign tumor arising from the meninges.",
    "pituitary": "Tumor affecting hormone-regulating pituitary gland.",
    "notumor": "No tumor detected in the scan."
}

# PDF REPORT GENERATOR
def generate_report(prediction, confidence, description):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("<b>NeuroScan AI - Diagnostic Report</b>", styles['Title']),
        Spacer(1, 20),
        Paragraph(f"<b>Prediction:</b> {prediction.upper()}", styles['Normal']),
        Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']),
        Spacer(1, 10),
        Paragraph(f"<b>Analysis:</b> {description}", styles['Normal']),
        Spacer(1, 20),
        Paragraph("Note: This is an AI-assisted result and not a medical diagnosis.", styles['Italic'])
    ]

    doc.build(content)
    buffer.seek(0)
    return buffer

# MODEL LOADING
@st.cache_resource
def load_model():
    base_model = timm.create_model("xception", pretrained=False, num_classes=0)

    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(base_model.num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 4)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model

model = load_model()

# FIND LAST CONV LAYER (for Grad-CAM)
def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer

# GRAD-CAM IMPLEMENTATION
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_index):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[0, class_index]
        loss.backward()

        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (299, 299))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

target_layer = get_last_conv_layer(model)
gradcam = GradCAM(model, target_layer)

# IMAGE PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512, 512), Image.LANCZOS)

    # Center the image nicely
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="MRI Scan", width=350)

    show_heatmap = st.checkbox("Show AI Attention Map", value=True)

    if st.button("Analyze Scan"):

        with st.spinner("Analyzing scan... please wait"):
            input_tensor = transform(image).unsqueeze(0)

            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).detach().numpy()[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = classes[predicted_index]
            confidence = probabilities[predicted_index] * 100

            cam = gradcam.generate(input_tensor, predicted_index)

        # RESULTS
        if predicted_class == "notumor":
            st.markdown('<div class="good">No Tumor Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bad">{predicted_class.upper()}</div>', unsafe_allow_html=True)

        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Insight:** {descriptions[predicted_class]}")

        # DOWNLOAD REPORT
        report = generate_report(predicted_class, confidence, descriptions[predicted_class])

        st.download_button(
            "Download Report",
            data=report,
            file_name="NeuroScan_Report.pdf",
            mime="application/pdf"
        )

        # PREDICTION CHART
        fig, ax = plt.subplots()
        ax.barh(classes, probabilities)
        ax.set_title("Prediction Breakdown")

        st.pyplot(fig)

        # HEATMAP (Grad-CAM)
        if show_heatmap:
            st.subheader("AI Attention Map")

            img_np = np.array(image.resize((299, 299)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img_np, caption="Original", width=280)

            with col2:
                st.image(overlay, caption="AI Focus", width=280)

# SIDEBAR
with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.markdown("---")

    st.markdown("### 🔍 Model Overview")
    st.markdown("""
    **Architecture:** Xception  
    **Input Size:** 299 × 299  
    **Classes:** 4  
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Features")
    st.markdown("""
    - MRI Scan Analysis  
    - Tumor Classification  
    - Confidence Score  
    - Attention Heatmap (Grad-CAM)  
    - Downloadable Report  
    """)

    st.markdown("---")

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    This tool is for educational purposes only.  
    It should not be used as a medical diagnosis.
    """)
