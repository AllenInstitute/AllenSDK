# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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
