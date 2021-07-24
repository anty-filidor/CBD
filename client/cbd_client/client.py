"""Includes set of functions for handling the CDB api."""
from urllib.parse import urljoin, urlparse

import requests  # type: ignore


def _uri_validator(url: str) -> bool:
    """Checks if given str is correct uri."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and " " not in url
    except ValueError:
        return False


def _prepare_failed_response(res: requests.models.Response) -> str:
    """Prepares human readable version of failed response."""
    return (
        f"Unsuccessful request! \n\t"
        f" error code: {str(res.status_code)}, \n\t"
        f" response content {str(res.content)}"
    )


class CBDClient:
    """Class to handle communication with the CBD api."""

    def __init__(self, url: str) -> None:
        """Initialises the object."""
        if not _uri_validator(url):
            raise ValueError(f"String {url} is not valid URL!")
        self.url = url

    # TODO
    def set_up_connection(self) -> None:
        """Check if app is accessible from given url in init."""
        pass

    def hello(self) -> str:
        """Gets the api's welcome message."""
        return requests.get(self.url).text

    def classify_string(self, text: str) -> str:
        """
        Performs classification on given string.

        The text will be handled by application and classified into one of three classes,
        non-harmful, cyberbullying, hate_speech

        :param text: the text to classify

        :return: human readable class that has been assigned to given text
        """
        _url = urljoin(self.url, "classify_string")
        res = requests.post(_url, json={"text": text})
        if res.ok:
            return res.json()["type"]
        else:
            return _prepare_failed_response(res)

    def get_model_info(self, prop: str) -> str:
        """
        Gets information about given parameter of the model.

        Function just ask api to print out details of the given parameter of the model.
        In most cases it doesn't exist so the response will not be 200.

        :param prop: property of model to ask api about

        :return: the string representation of value of the property
        """
        _url = urljoin(self.url, "get_model_info/")
        _url = urljoin(_url, prop)
        res = requests.get(_url)
        if res.ok:
            return res.json()["property"]
        else:
            return _prepare_failed_response(res)
