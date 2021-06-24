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


app = Flask(__name__)
cors = CORS(app)


def endpoint1(image_path):
    url = "https://resscvmodelapi-emekaborisama.cloud.okteto.net/spoof_real_torch"
    # Path to image fil
    filess = {"img": open(image_path, "rb")}
    results = requests.post(url, files=filess)
    #print("time taken:", time.time() - starttime)
    return(results.text)




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


        """basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)"""

        # Make prediction
        preds = endpoint1(file_path)
        return preds
    return None
