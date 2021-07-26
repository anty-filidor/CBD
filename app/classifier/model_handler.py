"""Contains functions to inference the model."""
import logging
import os
import random
from typing import Any, Callable, Dict
from unittest.mock import MagicMock

from . import data_processing

log = logging.getLogger(__name__)


def load_model(model_path: str) -> Callable:
    """Load model for further inference."""
    log.info(f"Path to model is {model_path}, exists? '{os.path.exists(model_path)}'.")
    return MagicMock()


@data_processing.convert_result
def inference(text: str, model: Any) -> int:
    """Perform classification of verified text on given model."""
    return random.randint(0, 2)


def classify_text(raw_data: Dict[str, Any], model: Callable) -> Dict[str, str]:
    """
    Process data obtained from HTTP request.

    :param raw_data: data obtained via flask app

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
