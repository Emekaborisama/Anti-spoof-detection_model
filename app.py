import streamlit as st
import sys
import time
import requests
import json
from PIL import Image, ImageOps
import os
here = os.path.dirname(__file__)
import base64
from tensorflow import keras
import numpy as np
np.random.seed(42)
import tensorflow as tf




def endpoint1(image_path):
    url = "https://resscvmodel-emekaborisama.cloud.okteto.net/spoof_real_torch"
    # Path to image fil
    filess = {"img": open(image_path, "rb")}
    results = requests.post(url, files=filess)
    #print("time taken:", time.time() - starttime)
    return(results.text)






html_temp = """
<div style = "background.color:teal; padding:10px">
<h2 style = "color:white; text_align:center;"> demo/h2>
<p style = "color:white; text_align:center;"> demo </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


#st.cache()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """





input_image= st.file_uploader("upload image or using your camera")

if input_image is not None:
    image = Image.open(input_image)
    im1 = image.save("geeks.jpg")
    st.image(image, caption='Uploaded Image.')
    image_path = "geeks.jpg"

if st.button("Classify"):
    tt = endpoint1(image_path = image_path)
    st.title("Prediction")
    st.markdown(tt)





