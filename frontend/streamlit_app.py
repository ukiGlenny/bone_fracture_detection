import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Bone Fracture Detection", page_icon="🦴", layout="wide")

st.title("Bone Fracture Detection System")

API_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                uploaded_file.seek(0)
                
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }
                
                response = requests.post(
                    f"{API_URL}/predict", 
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["num_detections"] > 0:
                        st.error(f"Found {result['num_detections']} fractures")
                        for i, det in enumerate(result["detections"], 1):
                            st.write(f"**{i}.** Confidence: {det['confidence']:.2%}")
                    else:
                        st.success("No fractures detected")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")