"""Contains functions to inference the model."""
import logging
import os
from typing import Any, Callable, Dict

import joblib
from sklearn.pipeline import Pipeline
from svm.dataset import Dataset
from svm.svm_classifier import tag_data

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
    dataset = Dataset(
        texts=[text], clean_data=False, remove_stopwords=False, is_train=False
    )
    tokenized_text = dataset.df["tokens"].values
    return tag_data(tokenized_text, model)[0]


def classify_text(raw_data: Dict[str, Any], model: Callable) -> Dict[str, str]:
    """
    Process data obtained from HTTP request.

    :param raw_data: data obtained via flask ml_pipeline

    :param model: the model for classification

    :return: dictionary with one item - 'type' filled by human readable name of predicted
            class for given text
    """
    log.info(f"Passed following data for classification {raw_data}")
    raw_string = data_processing.unpack_input_data(raw_data)
    text_for_classification = data_processing.anonimise_data(raw_string)

    pred_class = inference(text_for_classification, model)
    log.info(f"Predicted string as {pred_class}")

    return {"type": pred_class}
