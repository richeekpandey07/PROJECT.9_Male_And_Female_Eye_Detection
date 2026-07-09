import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="👁️ Eye Gender Detection AI",
    page_icon="👁️",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#B0B0B0;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:white;
    background: linear-gradient(135deg,#667eea,#764ba2);
    box-shadow:0px 5px 15px rgba(0,0,0,0.4);
}

.footer {
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<p class="title">👁️ Eye Gender Detection AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based Male vs Female Eye Classification</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
try:
    model = load_model("my_model.keras")
except Exception as e:
    st.error(f"Error Loading Model: {e}")
    st.stop()

# --------------------------------------------------
# Upload Image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # Preprocessing
    img = image.resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    score = float(prediction[0][0])

    if score > 0.5:
        result = "👨 MALE"
        confidence = score * 100
    else:
        result = "👩 FEMALE"
        confidence = (1 - score) * 100

    # Result Section
    with col2:

        st.markdown(
            f"""
            <div class="result-box">
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🎯 Confidence Score")

        progress_value = min(max(confidence / 100, 0.0), 1.0)
        st.progress(progress_value)

        st.metric(
            "Accuracy Confidence",
            f"{confidence:.2f}%"
        )

        st.success("Prediction Completed Successfully!")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.markdown("""
### 🚀 About Project

This AI model predicts gender from eye images using Deep Learning techniques.

### 👨‍💻 Developer

**Richeek Pandey**

🔗 GitHub: https://github.com/richeekpandey07

🔗 LinkedIn: https://www.linkedin.com/in/richeek-pandey

⭐ If you like this project, don't forget to star the repository!
""")
