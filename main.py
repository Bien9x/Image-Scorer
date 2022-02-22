import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from PIL import Image
import albumentations
from torchvision import transforms
from model import AestheticModel
from utils import sigmoid

import config

model = AestheticModel()

transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


@st.cache
def load_model(path):
    model.load(path, device="cpu")
    model.eval()


with st.spinner("Loading model..."):
    load_model("cp/basic_swin_small.bin")


def predict(img):
    img = transforms(img).unsqueeze(0)
    logit = model(img)[0].item()
    return sigmoid(logit)


uploaded_files = st.file_uploader("Choose image files", type=['jpeg', 'jpg', 'png'], accept_multiple_files=True)

pil_images = []
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    col1, col2 = st.columns(2)
    image = Image.open(BytesIO(bytes_data)).convert('RGB')
    with col1:
        st.image(bytes_data)
        st.write("filename:", uploaded_file.name)

    with col2:
        score = predict(image)
        st.subheader("{0:.0%}".format(score))
        st.text("chance that this image is awesome ")
