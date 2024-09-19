import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("sar-colorizer\pix2pix_generator (4).h5")
    return model

def colorize_image(model, bw_image):
    bw_image = bw_image.resize((256, 256))  
    bw_image = np.array(bw_image) / 255.0  
    bw_image = np.expand_dims(bw_image, axis=0)  
    colorized_image = model.predict(bw_image)
    colorized_image = np.squeeze(colorized_image, axis=0)  
    colorized_image = (colorized_image * 255).astype(np.uint8)
    return Image.fromarray(colorized_image)

model = load_model()

st.title("SAR Image Colorization")

uploaded_file = st.file_uploader("Upload a black and white SAR image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    bw_image = Image.open(uploaded_file).convert("L")
    st.image(bw_image, caption="Uploaded Image", use_column_width=True)
    colorized_image = colorize_image(model, bw_image)
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
