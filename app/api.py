from __future__ import division, print_function
# coding=utf-8
import os
from PIL import Image, ImageOps
import requests
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS, cross_origin
import cv2

app = Flask(__name__)
cors = CORS(app)


def endpoint1(image_path):
    url = "https://resscvmodelapi-emekaborisama.cloud.okteto.net/spoof_real_tensor1"
    # Path to image fil
    
    filess = {"img": open(image_path, "rb")}
    results = requests.post(url, files=filess)
    #print("time taken:", time.time() - starttime)
    return(results.text)

def crop_Casa(imagepath):
    img = cv2.imread(imagepath)
    # Convert into grayscal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
	    faces = img[y:y + h, x:x + w]
	    #cv2.imshow("img", faces)
	    cv2.imwrite('face.jpg', img)
    
    return

@app.route('/Image_Classification', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        image = Image.open(f)
        im1 = image.save("geeks.jpg")
        file_path = "geeks.jpg"
        file_path2= "face.jpg"


        """basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)"""

        # Make prediction
        crop_Casa(file_path)
        preds = endpoint1(file_path)
        return preds
    return None
