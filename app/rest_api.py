"""Contains Flask API for CBD model inference."""
import os
import sys
from unittest.mock import MagicMock

from pathlib import Path
import logging
import flask
from flask import Flask, jsonify, request
from utils import prepare_gpu, init_logger

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

init_logger(Path("."))
log = logging.getLogger(__name__)

log.info("loading CBD model")
mocky_model = MagicMock()

log.info("preparing gpu")
prepare_gpu()

log.info("creating flask app")
app = Flask(__name__)


@app.route("/")
def hello() -> str:
    """
    Print out message to see that app works.

    :return: hello message
    """
    return "Hello I'm Cyberbullying Detector!"


@app.route("/classify_string/<uuid>", methods=["GET", "POST"])
def classify_string(uuid: str) -> flask.Response:
    """
    Run model on given string to check whether it is dangerous.

    :param uuid: uuid of the response

    :return: str representation of class: "non-harmful", "cyberbullying", "hate-speech"
    """
    content = request.json
    log.info(content)
    log.warning(MODEL_PATH_LDA, MODEL_PATH_NN)
    log.info(sys.version)
    log.info(os.getcwd())

    return jsonify({"uuid": uuid})


@app.route("/get_model_info/<prop>")
def get_model_info(prop: str) -> flask.Response:
    """
    Get info about the CBD model's property.

    :param prop: property of the model, i.e. its attribute

    :return: dict with value of property
    """
    return jsonify({"property": str(mocky_model.__dict__.get(prop))})


if __name__ == "__main__":
    print("Starting app")
    app.run(debug=True, host="0.0.0.0")
