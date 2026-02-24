import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# Page configuration
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and generate an AI-powered caption instantly.")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

captioner = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            result = captioner(image)
            caption = result[0]["generated_text"]

        st.success("Caption Generated!")
        st.subheader("📝 Caption:")
        st.write(caption)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Hugging Face Transformers")
