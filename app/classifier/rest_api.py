"""Contains Flask API for CBD model inference."""
import logging
import os
from pathlib import Path

import flask
from flask import Flask, jsonify, request

# turn off tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from . import model_handler  # isort:skip  # noqa: E402
from . import utils  # isort:skip  # noqa: E402

utils.init_logger(Path("."))
log = logging.getLogger(__name__)

log.info("Preparing GPU or CPU for Tensorflow")
utils.prepare_gpu()

log.info("Loading CBD model")
model = model_handler.load_model(os.environ["MODEL_PATH"])

log.info("Creating Flask application")
app = Flask(__name__)


@app.route("/")
def hello() -> str:
    """
    Print out message to see that app works.

    :return: hello message
    """
    return "Hello I'm Cyberbullying Detector!"


@app.route("/classify_string", methods=["GET", "POST"])
def classify_string() -> flask.Response:
    """
    Run model on given string to check whether it is dangerous.

    :return: str representation of class: "non-harmful", "cyberbullying", "hate-speech"
    """
    return jsonify(model_handler.classify_text(request.json, model))


@app.route("/get_model_info/<prop>")
def get_model_info(prop: str) -> flask.Response:
    """
    Get info about the CBD model's property.

    :param prop: property of the model, i.e. its attribute

    :return: dict with value of property
    """
    return jsonify({"property": str(model.__dict__.get(prop))})
