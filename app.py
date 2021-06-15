import streamlit as st
import sys
import time
import requests
import json
from PIL import Image






def endpoint1(text):
    url = "https://ilistener2-emekaborisama.cloud.okteto.net/named_entity"
    payload={'text': text}
    files=[

    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return (response.text)



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




