# We now need the json library so we can load and export json data
import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from unittest.mock import MagicMock

from flask import Flask, request, jsonify

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)


# init logger


# load model
mocky_model = MagicMock()

# create flask app
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello I'm Cyberbullying Detector!"


@app.route('/classify_post/<uuid>', methods=['GET', 'POST'])
def classify_post(uuid):
    content = request.json
    print(content)
    return jsonify({"uuid": uuid})


@app.route('/get_model_info/<prop>')
def get_model_info(prop):
    return {"property": str(mocky_model.__dict__[property])}


if __name__ == "__main__":
    print("Starting app")
    app.run(debug=True, host='0.0.0.0')

