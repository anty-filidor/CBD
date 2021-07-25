from unittest.mock import MagicMock, patch

import pytest
from cbd_client.client import CBDClient, _prepare_failed_response, _uri_validator


@pytest.mark.parametrize(
    "uri, exp_result",
    [
        ("http://www.google.pl", True),
        ("http://192.168.0.1:4000", True),
        ("hahaha", False),
        ("http://www.google.pl 33 1", False),
    ],
)
def test__uri_validator(uri, exp_result):
    real_result = _uri_validator(uri)
    assert exp_result == real_result


@patch("requests.models.Response")
def test__prepare_failed_response(mo_resp):
    resp = mo_resp()
    resp.status_code = 404
    resp.content = "dummy content"

    exp_res = (
        f"Unsuccessful request! \n\t error code: {resp.status_code}, \n\t "
        f"response content {resp.content}"
    )

    assert exp_res == _prepare_failed_response(resp)


class TestCBDClient:
    def test_check_connection(self):
        # TODO
        # here we need do mock urllib.request.urlopen to return proper code and check it
        # has been called once. We can parametrize this test to cover if statement
        pass

    @patch("requests.get", new_callable=MagicMock())
    @patch("cbd_client.client._uri_validator", new=MagicMock(return_value=True))
    def test_hello(self, mo_get):
        url = "dummy url"

        cbd_client = CBDClient(url)
        cbd_client.hello()

        mo_get.assert_called_with(url)
        mo_get.assert_called_once()

    def test_classify_string(self):
        # TODO
        # here we need do mock requests.post and with proper parameters and check it has
        # been called once. We can also parametrize this test to cover if statement
        pass

    def test_get_model_info(self):
        # TODO
        # here we need do mock requests.get and with proper url address and check it has
        # been called once. We can also parametrize this test to cover if statement.
        pass
