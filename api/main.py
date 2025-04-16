import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ==============================
# Model Architecture Definition (Must match training code)
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
# PyTorch Model Prediction Function
# ==============================
def model_prediction(test_image):
    # Load model (update path as needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WheatDiseaseModel(num_classes=5)
    model.load_state_dict(torch.load("./wheat_disease_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Image preprocessing (must match validation transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, class_logits = model(input_tensor)
        probs = torch.nn.functional.softmax(class_logits, dim=1)
    
    return torch.argmax(probs).item()

# ==============================
# Streamlit UI
# ==============================
st.header("Wheat Disease Recognition")

# Upload an image
test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

if test_image is not None:
    # Option to display the uploaded image
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Predict the disease
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_names = ["Brown_rust", "Healthy", "Loose_Smut", "Yellow_rust", "septoria"]
        st.success(f"Model Prediction: {class_names[result_index]}")
else:
    st.info("Please upload an image to begin.")