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


def test_write_integer_array(dict_obj):
    s_in = ju.write_string({ "int_array": dict_obj["int_array"] })
    s_out = """{
  "int_array": [
    [
      0,
      1
    ],
    [
      0,
      1
    ]
  ]
}"""

    assert s_in == s_out

def test_write_float_array(dict_obj):
    s_in = ju.write_string({ "float_array": dict_obj["float_array"] })
    s_out ="""{
  "float_array": [
    [
      0.0,
      0.0
    ],
    [
      0.23809523809523808,
      0.23809523809523808
    ]
  ]
}"""

    assert s_in == s_out

def test_write_string(dict_obj):
    s_in = ju.write_string({ "string": dict_obj["string"] })
    s_out = """{
  "string": "test string"
}"""

    assert s_in == s_out

def test_write_bool_array(dict_obj):
    s_in = ju.write_string({ "bool_array": dict_obj["bool_array"] })
    s_out = """{
  "bool_array": [
    [
      false,
      true
    ],
    [
      false,
      true
    ]
  ]
}"""
    assert s_in == s_out

def test_write_list(dict_obj):
    s_in = ju.write_string({ "list": dict_obj["list"] })
    s_out = """{
  "list": [
    "this",
    "is",
    1,
    "list"
  ]
}"""
    assert s_in == s_out
