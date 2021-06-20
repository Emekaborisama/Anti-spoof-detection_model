import cv2 as cv
import streamlit as st
import numpy as np
from app import load_pred
from pytorchpred import load_predict

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv.VideoCapture(0)

while run:
    _, frame = camera.read()
    FRAME_WINDOW.image(frame)
    cv.imwrite(filename="alpha.png", img=frame)
    image_result = 'alpha.png'
    model_result = load_predict(image_result)
    st.title("Real or Spoof")
    st.write(model_result)
    


else:
    st.write('Stopped')