import allensdk.core.json_utilities as ju
import pytest
from mock import patch, MagicMock, call
import numpy as np


@pytest.fixture
def dict_obj():
    int_array, y = np.meshgrid(np.arange(2), np.arange(2))
    float_array = y.astype(float)/4.2
    bool_array =  int_array > 0

    object = {"string": "test string",
              "float_array": float_array,
              "int_array": int_array,
              "bool_array": bool_array,
              "list": ["this", "is", 1, "list"]}
    return object


def test_write_string(dict_obj):
    ju.write_string(dict_obj)
    with patch("allensdk.core.json_utilities.json_handler",
               MagicMock()) as json_handler:
        ju.write_string(dict_obj)
        expected = [call(False), call(True), call(False), call(True)]
        assert json_handler.call_args_list == expected
