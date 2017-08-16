# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2015-2016. Allen Institute. All rights reserved.
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
import pytest
from mock import patch, mock_open, Mock
from simplejson.scanner import JSONDecodeError
import allensdk.core.json_utilities as ju
from allensdk.core.json_utilities import JsonComments
import logging
try:
    import __builtin__ as builtins  # @UnresolvedImport
except:
    import builtins  # @UnresolvedImport


@pytest.fixture
def commented_json():
    return ("{\n"
            "    // comment\n"
            "    \"color\": \"blue\"\n"
            "}")


@pytest.fixture
def blank_line_json():
    return ("{\n"
            "\n"
            "\n"
            "\n"
            "    \"color\": \"blue\"\n"
            "}")


@pytest.fixture
def multi_line_json():
    return ("{\n"
            "/* \n"
            " * multiline comment\n"
            " */\n"
            "    \"color\": \"blue\"\n"
            "}")


@pytest.fixture
def two_multi_line_json():
    return ("{\n"
            "    \"colors\": [\"blue\",\n"
            "    /* comment these out\n"
            "    \"red\",\n"
            "    \"yellow\",\n"
            "     ... but not these */\n"
            "    \"orange\",\n"
            "    \"purple\",\n"
            "    /* also comment this out\n"
            "    \"indigo\",\n"
            "    .... end comment */\n"
            "    \"violet\"\n"
            "    ]\n"
            "}")


@pytest.fixture
def corrupted_json():
    return ("{\n"
            "    \"colors\": \"blue\",\n"
            "    /* comment these out\n"
            "    \"red\",\n"
            "    \"yel")


@pytest.fixture
def ju_logger():
    log = logging.getLogger('allensdk.core.json_utilities')
    log.error = Mock()

    return log


def testSingleLineCommentJSONDecodeError(corrupted_json,
                                    ju_logger):
    with pytest.raises(JSONDecodeError) as e_info:
        with patch(builtins.__name__ + ".open",
                   mock_open(read_data=corrupted_json)):
            JsonComments.read_file("corrupted.json")

    ju_logger.error.assert_called_once_with(
        'Could not load json object from file: corrupted.json')
    assert e_info.typename == 'JSONDecodeError'


def testSingleLineComment(commented_json):
    parsed_json = JsonComments.read_string(
        commented_json)

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testBlankLines(blank_line_json):
    parsed_json = JsonComments.read_string(
        blank_line_json)

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testMultiLineComment(multi_line_json):
    parsed_json = JsonComments.read_string(
        multi_line_json)

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testTwoMultiLineComments(two_multi_line_json):
    parsed_json = JsonComments.read_string(
        two_multi_line_json)

    assert('colors' in parsed_json)
    assert(len(parsed_json['colors']) == 4)
    assert('blue' in parsed_json['colors'])
    assert('orange' in parsed_json['colors'])
    assert('purple' in parsed_json['colors'])
    assert('violet' in parsed_json['colors'])


def testSingleLineCommentFile(commented_json):
    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=commented_json)):
        parsed_json = JsonComments.read_file('mock.json')

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testBlankLinesFile(blank_line_json):
    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=blank_line_json)):
        parsed_json = JsonComments.read_file('mock.json')

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testMultiLineFile(multi_line_json):
    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=multi_line_json)):
        parsed_json = JsonComments.read_file('mock.json')

    assert('color' in parsed_json and
           parsed_json['color'] == 'blue')


def testTwoMultiLineFile(two_multi_line_json):
    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=two_multi_line_json)):
        parsed_json = JsonComments.read_file('mock.json')

    assert('colors' in parsed_json)
    assert(len(parsed_json['colors']) == 4)
    assert('blue' in parsed_json['colors'])
    assert('orange' in parsed_json['colors'])
    assert('purple' in parsed_json['colors'])
    assert('violet' in parsed_json['colors'])


def test_write_nan():
    with patch(builtins.__name__ + ".open",
               mock_open(),
               create=True) as mo:
        ju.write('/some/file/test.json', { "thing": float('nan')})
    
    assert 'null' in str(mo().write.call_args_list[0])
