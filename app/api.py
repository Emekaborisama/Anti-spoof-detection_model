from flask import Flask, request, jsonify, Response
from flask import render_template, url_for, redirect, flash
#from spacy import displacy
from app.pytorchpred import load_predict
from flask_cors import CORS, cross_origin
import os
here = os.path.dirname(__file__)
imagepath = os.path.join(here, 'img.jpg')


app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def home():
    return("welcome to api")


@app.route("/spoof_real_torch", methods=['POST'])
def real():
    data = request.files["img"]
    data.save("./app/img.jpg")
    modelresult = load_predict(imagepath)
    print(modelresult)
    return modelresult





