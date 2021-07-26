"""Contains functions to inference the model."""
import logging
import os
import random
from typing import Any, Callable, Dict

import joblib
from ml.dataset import Dataset
from ml.svm_classifier import tag_data
from sklearn.pipeline import Pipeline

from . import data_processing

log = logging.getLogger(__name__)


def load_model(model_path: str) -> Pipeline:
    """Load model for further inference."""
    log.info(f"Path to model is {model_path}, exists? '{os.path.exists(model_path)}'.")
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model


@data_processing.convert_result
def inference(text: str, model: Pipeline) -> int:
    """Perform classification of verified text on given model."""
    text = Dataset(texts=[text], clean_data=True, remove_stopwords=True, is_train=False)
    tokenized_text = text.df["tokens"].values
    return tag_data(tokenized_text, model)[0]


def classify_text(raw_data: Any, model: Callable) -> Dict[str, str]:
    """
    Process data obtained from HTTP request.

    :param raw_data: data obtained via flask app

    :param model: the model for classification

    :return: dictionary with one item - 'type' filled by human readable name of predicted
            class for given text
    """
    log.info(f"Passed following data for classification {raw_data}")
    text_for_classification = data_processing.unpack_input_data(raw_data)

    pred_class = inference(text_for_classification, model)
    log.info(f"Predicted string as {pred_class}")

    return {"type": pred_class}
