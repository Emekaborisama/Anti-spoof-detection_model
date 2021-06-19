from flask import Flask, request, jsonify, Response
from flask import render_template, url_for, redirect, flash
#from spacy import displacy
from pytorchpred import load_predict
from app.loc_extract import extractloc
from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def home():
    return("welcome to api")


@app.route("/spoof_real_torch", methods=['POST'])
def nerdy():
    imagepath = request.form['path']
    modelresult = load_predict(imagepath)
    return modelresult





