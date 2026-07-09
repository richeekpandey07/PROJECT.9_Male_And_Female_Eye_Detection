import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="👁️ Eye Gender Detection AI",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
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

# Header
st.markdown(
    '<p class="title">👁️ Eye Gender Detection AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based Male vs Female Eye Classification</p>',
    unsafe_allow_html=True
)

# Load model
model = load_model("my_model.keras")

# Upload
uploaded_file = st.file_uploader(
    "📤 Upload Eye Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((299,299))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "👨 MALE"
        confidence = prediction[0][0] * 100
    else:
        result = "👩 FEMALE"
        confidence = (1-prediction[0][0]) * 100

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
        st.progress(confidence/100)

        st.metric(
            "Accuracy Confidence",
            f"{confidence:.2f}%"
        )

        st.success(
            f"Prediction Completed Successfully!"
        )

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
