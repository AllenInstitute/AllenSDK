# Copyright 2015-2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


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
