import streamlit as st
import sys
import time
import requests
import json
from PIL import Image, ImageOps
import os
here = os.path.dirname(__file__)

from tensorflow import keras
import numpy as np
np.random.seed(42)
import tensorflow as tf

modelpath= os.path.join(here, 'mob_netmodel_saved.h5')
def load_pred(image_inf):
    model2 = keras.models.load_model(modelpath, compile=False)
    #image = tf.keras.preprocessing.image.load_img(image_inf, target_size= (96, 96, 3))
    image = ImageOps.fit(image_inf, size =(96, 96))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    preds = np.argmax(predictions[0])
    if preds == 1:
        return("real")
    elif preds == 0:
        return("pls retake the profile image")








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
    st.image(image, caption='Uploaded Image.', use_column_width=True)

if st.button("Classify"):
    tt = load_pred(image_inf = image)
    st.title("Prediction")
    st.markdown(tt)





