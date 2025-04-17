import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Wheat Leaf Idaentifier",
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
    /* Global Styles */
    [data-testid="stAppViewContainer"] {{
        background-color: #FFFFFF;
    }}
    
    /* Header Banner */
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
    
    /* Banner Content */
    .banner-content {{
        position: relative;
        z-index: 2;
        flex: none;
        width: 52%;
        padding: 0.8rem 0.5rem 0.8rem 1.2rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 220px;
    }}
    
    /* Title Container */
    .title-container {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: 100%;
        padding: 0.7rem 0;
    }}
    
    .title-text {{
        color: #FFFFFF;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        line-height: 1.2;
        width: 100%;
        max-width: 100%;
    }}
    
    .title-text h1 {{
        font-size: 2.3em;
        font-weight: 800;
        margin: 0;
        padding: 0;
        letter-spacing: -0.01em;
        white-space: nowrap;
        overflow: visible;
        width: 100%;
        display: block;
    }}
    
    .title-text h2 {{
        font-size: 1.8em;
        font-weight: 600;
        margin: 0;
        padding: 0;
        letter-spacing: 0;
        opacity: 0.95;
        white-space: nowrap;
        overflow: visible;
        width: 100%;
        display: block;
    }}
    
    /* Wheat Image Container */
    .wheat-image-wrapper {{
        position: absolute;
        right: -15px;
        top: 50%;
        transform: translateY(-50%);
        height: 100%;
        width: 48%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        overflow: visible;
    }}
    
    .wheat-image {{
        position: absolute;
        right: 0;
        height: 180%;
        width: 220px;
        background-image: url('{wheat_image if wheat_image else ""}');
        background-size: contain;
        background-position: center right;
        background-repeat: no-repeat;
        transform-origin: center;
        transform: translateY(0) scale(1.15);
        image-rendering: -webkit-optimize-contrast;
        opacity: 1;
        filter: contrast(1.05) brightness(1.02);
    }}
    
    /* Disease Result Styling */
    .result-container {{
        margin: 1.5rem 0;
        padding: 1.2rem;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .disease-result {{
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0;
        font-weight: 600;
        font-size: 1.3em;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .disease-warning {{
        background: linear-gradient(135deg, #FFE4E4 0%, #FFD0D0 100%);
        color: #D43B3B;
        border: 1px solid rgba(212, 59, 59, 0.3);
        box-shadow: 0 2px 12px rgba(212, 59, 59, 0.15);
        animation: pulseWarning 2s infinite;
    }}
    
    .disease-healthy {{
        background: linear-gradient(135deg, #E6FFE6 0%, #98FB98 100%);
        color: #228B22;
        border: 1px solid rgba(34, 139, 34, 0.3);
        box-shadow: 0 2px 12px rgba(34, 139, 34, 0.15);
    }}

    @keyframes pulseWarning {{
        0% {{
            transform: scale(1);
            box-shadow: 0 2px 12px rgba(212, 59, 59, 0.15);
        }}
        50% {{
            transform: scale(1.02);
            box-shadow: 0 4px 16px rgba(212, 59, 59, 0.25);
        }}
        100% {{
            transform: scale(1);
            box-shadow: 0 2px 12px rgba(212, 59, 59, 0.15);
        }}
    }}

    /* Disease Details Container */
    .disease-details {{
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 8px;
        background: rgba(255, 228, 228, 0.3);
        border: 1px solid rgba(212, 59, 59, 0.2);
    }}

    .disease-details h3 {{
        color: #D43B3B;
        margin-bottom: 0.5rem;
        font-size: 1.1em;
        font-weight: 600;
    }}

    .disease-details p {{
        color: #333333;
        margin: 0.5rem 0;
        line-height: 1.4;
        font-size: 1em;
    }}
    
    /* Button Styling */
    .stButton > button,
    .stButton > button *,
    .stButton button span {{
        background-color: #92C756 !important;
        color: white !important;
        font-size: 14px !important;
        padding: 16px 24px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        margin: 1px 0 !important;
        transition: all 0.3s ease !important;
        font-family: "Times New Roman", Times, serif !important;
        letter-spacing: 0 !important;
        text-rendering: optimizeLegibility !important;
        -webkit-font-smoothing: antialiased !important;
        line-height: 1 !important;
        font-weight: 500 !important;
        height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        text-align: center !important;
        box-sizing: border-box !important;
    }}
    
    .stButton > button:hover {{
        background-color: #7DAD48 !important;
        border: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(146, 199, 86, 0.2) !important;
    }}

    /* Additional button text override */
    .stButton button[kind="secondary"] span,
    .stButton button[kind="primary"] span {{
        font-family: "Times New Roman", Times, serif !important;
        font-size: 14px !important;
        letter-spacing: 0 !important;
    }}
    
    /* Center Section */
    .center-section {{
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }}
    
    .leaf-icon {{
        color: #92C756;
        font-size: 2.2em;
        margin-bottom: 0.5rem;
    }}
    
    .subtitle {{
        color: #4A4A4A;
        font-size: 1.1em;
        line-height: 1.5;
        margin: 0.5rem 0 2.5rem 0;
        font-weight: 500;
    }}
    
    /* Button Container */
    .stButton {{
        margin-top: 0.2rem !important;
    }}
    
    .stButton:first-child {{
        margin-bottom: 0.1rem !important;
    }}
    
    .stButton:last-child {{
        margin-top: 0.1rem !important;
    }}
    
    /* Hide default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .header-banner {{
            padding: 2.5rem 1rem;
            min-height: 170px;
        }}
        .banner-content {{
            width: 50%;
            padding: 0.7rem 0.4rem 0.7rem 1rem;
            min-width: 180px;
        }}
        .title-container {{
            gap: 8px;
            padding: 0.6rem 0;
        }}
        .title-text h1 {{
            font-size: 2em;
        }}
        .title-text h2 {{
            font-size: 1.5em;
        }}
        .wheat-image-wrapper {{
            right: -12px;
            width: 50%;
        }}
        .wheat-image {{
            width: 195px;
            height: 175%;
            transform: translateY(0) scale(1.12);
        }}
    }}

    @media (max-width: 480px) {{
        .header-banner {{
            padding: 2.3rem 0.8rem;
            min-height: 160px;
        }}
        .banner-content {{
            width: 52%;
            padding: 0.6rem 0.3rem 0.6rem 0.8rem;
            min-width: 150px;
        }}
        .title-container {{
            gap: 7px;
            padding: 0.5rem 0;
        }}
        .title-text h1 {{
            font-size: 1.7em;
        }}
        .title-text h2 {{
            font-size: 1.35em;
        }}
        .wheat-image-wrapper {{
            right: -10px;
            width: 48%;
        }}
        .wheat-image {{
            width: 170px;
            height: 170%;
            transform: translateY(0) scale(1.1);
        }}
    }}

    /* Global Font Style */
    [data-testid="stAppViewContainer"],
    .stMarkdown,
    .stButton > button,
    .stText,
    div[data-testid="stText"],
    div[data-testid="stMarkdown"],
    .element-container,
    .header-banner,
    .title-container,
    .title-text,
    .title-text h1,
    .title-text h2,
    .center-section,
    .subtitle,
    .disease-result,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stCamera"] div,
    [data-testid="stCamera"] span {{
        font-family: "Times New Roman", Times, serif !important;
    }}
</style>
""", unsafe_allow_html=True)

# TensorFlow Model Functions
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./wheat.h5")

def model_prediction(image_data):
    model = load_model()
    image = Image.open(image_data)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

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
                    st.markdown('</div>', unsafe_allow_html=True)