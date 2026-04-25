
import streamlit as st
import cv2, numpy as np, json, tempfile, os
from PIL import Image
import sys
sys.path.insert(0, '/content')

st.set_page_config(page_title="KnightSight ANPR", page_icon="🚗", layout="wide")
st.title("🚗 KnightSight EdgeVision — ANPR System")
st.markdown("**Pipeline:** YOLOv8n → YOLOv8-plate → 7-variant OCR → Indian plate validator")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("Running ANPR pipeline..."):
        # Import pipeline from notebook (run notebook cells first)
        try:
            from __main__ import process_image, draw_results
            annotated, v_boxes, plate_results = process_image(tmp_path, verbose=True)
        except ImportError:
            st.error("Pipeline not loaded — run all notebook cells first, then launch this app.")
            st.stop()

    with col2:
        st.subheader("Annotated Output")
        if annotated is not None:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("Detected Plates")
    valid = [r for r in plate_results if r.get("plate_number")]
    if valid:
        for r in valid:
            badge = "✅" if r.get("is_valid_plate") else "⚠️"
            st.write(f"{badge} **{r['plate_number']}** — OCR conf: {r['ocr_confidence']:.2f} | Vehicle: {r.get('vehicle_class','-')}")
    else:
        st.warning("No plate text detected.")

    st.subheader("JSON Output (structured)")
    st.json(plate_results)
    os.unlink(tmp_path)
