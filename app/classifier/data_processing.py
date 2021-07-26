"""Contains functions for data processing."""

from functools import wraps
from typing import Any, Callable


def unpack_input_data(data: Any) -> str:
    """
    Unpack input data from initial dict to string.

    :param data: the raw dict obtained from http request

    :return: string that can be consumed by the model
    """
    text = data.get("text")
    if isinstance(text, str) and len(text) > 0:
        return text
    else:
        raise AttributeError(
            f"Input data {data} has no field called 'text' or that field is empty!"
        )


def anonimise_data(raw_string: str) -> str:
    """TODO - write finc that converts all tags to @anonimized_account."""
    pass


def convert_result(func: Callable) -> Callable:
    """
    Decorate model inference function by converting result to string.

    :param func: model inference function

    :return: decorated function
    """

    @wraps(func)
    def _func_wrapper(*args: Any, **kwargs: Any) -> str:
        result_int = func(*args, **kwargs)
        if result_int == 0:
            return "non-harmful"
        elif result_int == 1:
            return "cyberbullying"
        elif result_int == 2:
            return "hate_speech"
        else:
            raise ValueError(
                f"Model should return int in range [0, 2]. {result_int} is not valid!"
            )

    return _func_wrapper
