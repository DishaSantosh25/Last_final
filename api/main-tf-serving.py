import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64

# ==============================
# Model Architecture Definition
# ==============================
import timm
import torch.nn as nn

class WheatDiseaseModel(nn.Module):
    def __init__(self, num_classes=5):
        super(WheatDiseaseModel, self).__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, features_only=True)
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)[-1]
        seg_mask = self.segmentation_head(features)
        pooled = self.avgpool(features).flatten(1)
        class_logits = self.classifier(pooled)
        return seg_mask, class_logits

# ==============================
# PyTorch Model Loading
# ==============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WheatDiseaseModel(num_classes=5)
    model.load_state_dict(torch.load("./wheat_disease_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# ==============================
# Prediction Function
# ==============================
def model_prediction(image_data):
    model = load_model()
    device = next(model.parameters()).device
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_data).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, class_logits = model(input_tensor)
        probs = torch.nn.functional.softmax(class_logits, dim=1)
    
    return torch.argmax(probs).item()

# ==============================
# Streamlit UI (Original)
# ==============================
# Page configuration
st.set_page_config(
    page_title="Wheat Leaf Identifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load and encode the wheat image
def get_wheat_image():
    try:
        with open("extracted_wheat.png", "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error loading wheat image: {e}")
        try:
            with open("wheat.jpg", "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            print(f"Error loading fallback image: {e}")
            return None

wheat_image = get_wheat_image()

# Custom CSS with dynamic image
st.markdown(f"""
<style>
    /* Original CSS styles remain unchanged */
    [data-testid="stAppViewContainer"] {{
        background-color: #FFFFFF;
    }}
    .header-banner {{
        background: linear-gradient(135deg, #F5C06B 0%, #F9D69B 100%);
        border-radius: 16px;
        padding: 2.8rem 1.2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: visible;
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        box-shadow: 0 4px 15px rgba(245, 192, 107, 0.2);
    }}
    /* ... (keep all original CSS styles exactly as they were) ... */
</style>
""", unsafe_allow_html=True)

# Header Banner with Wheat Image
st.markdown("""
    <div class="header-banner">
        <div class="banner-content">
            <div class="title-container">
                <div class="title-text">
                    <h1>Wheat Leaf</h1>
                    <h2>Identifier</h2>
                </div>
            </div>
        </div>
        <div class="wheat-image-wrapper">
            <div class="wheat-image" role="img" aria-label="Decorative wheat image"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Center Section
st.markdown("""
    <div class="center-section">
        <div class="leaf-icon">🌿</div>
        <div class="subtitle">
            Supporting Farmers in<br>Safeguarding their Crop Health
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = None

# Create columns for better layout
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Direct button implementation
    camera_btn = st.button("📸 Take picture of your plant")
    gallery_btn = st.button("🖼️ Import from your gallery")

    # Handle button clicks
    if camera_btn:
        st.session_state.current_view = 'camera'
    if gallery_btn:
        st.session_state.current_view = 'gallery'

    # Show appropriate view based on button clicks
    if st.session_state.current_view == 'camera':
        camera_input = st.camera_input("")
        if camera_input:
            st.image(camera_input)
            if st.button("Analyze Image", key="analyze_camera"):
                with st.spinner("📊 Analyzing your wheat leaf..."):
                    result_index = model_prediction(camera_input)
                    class_names = ["Brown_rust", "Healthy", "Loose_Smut", "Yellow_rust", "septoria"]
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    if class_names[result_index] == "Healthy":
                        st.markdown(
                            f'<div class="disease-result disease-healthy">✨ Your wheat plant is healthy!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        disease_name = class_names[result_index]
                        recommendations = {
                            "Brown_rust": "Apply fungicide treatment immediately. Monitor other plants for early signs of infection.",
                            "Loose_Smut": "Remove and destroy infected plants. Consider using disease-resistant wheat varieties for future planting.",
                            "Yellow_rust": "Apply appropriate fungicides. Improve air circulation between plants and reduce humidity.",
                            "septoria": "Use foliar fungicides. Maintain proper spacing between plants to reduce moisture."
                        }
                        st.markdown(
                            f'''
                            <div class="disease-result disease-warning">⚠️ Disease Detected: {disease_name}</div>
                            <div class="disease-details">
                                <h3>About {disease_name}</h3>
                                <p>{recommendations.get(disease_name, "This wheat disease requires immediate attention. Early detection allows for effective treatment and prevents spread to other plants.")}</p>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_view == 'gallery':
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            st.image(uploaded_file)
            if st.button("Analyze Image", key="analyze_upload"):
                with st.spinner("📊 Analyzing your wheat leaf..."):
                    result_index = model_prediction(uploaded_file)
                    class_names = ["Brown_rust", "Healthy", "Loose_Smut", "Yellow_rust", "septoria"]
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    if class_names[result_index] == "Healthy":
                        st.markdown(
                            f'<div class="disease-result disease-healthy">✨ Your wheat plant is healthy!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        disease_name = class_names[result_index]
                        recommendations = {
                            "Brown_rust": "Apply fungicide treatment immediately. Monitor other plants for early signs of infection.",
                            "Loose_Smut": "Remove and destroy infected plants. Consider using disease-resistant wheat varieties for future planting.",
                            "Yellow_rust": "Apply appropriate fungicides. Improve air circulation between plants and reduce humidity.",
                            "septoria": "Use foliar fungicides. Maintain proper spacing between plants to reduce moisture."
                        }
                        st.markdown(
                            f'''
                            <div class="disease-result disease-warning">⚠️ Disease Detected: {disease_name}</div>
                            <div class="disease-details">
                                <h3>About {disease_name}</h3>
                                <p>{recommendations.get(disease_name, "This wheat disease requires immediate attention. Early detection allows for effective treatment and prevents spread to other plants.")}</p>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)