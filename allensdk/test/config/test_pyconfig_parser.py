# Copyright 2015 Allen Institute for Brain Science
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
from mock import patch, mock_open
from allensdk.config.model.description_parser import DescriptionParser
try:
    import __builtin__ as builtins
except:
    import builtins


@pytest.fixture
def pyconfig():
    file_1 = ("{\n"
              "    \"section_A\": [\n"
              "       {\n"
              "           \"prop_a\": \"val_a\",\n"
              "           \"prop_b\": \"val_b\"\n"
              "       },\n"
              "       {\n"
              "           \"prop_c\": \"val_c\",\n"
              "           \"prop_d\": \"val_d\"\n"
              "       }\n"
              "    ],\n"
              "    \"section_B\": [\n"
              "       {\n"
              "           \"prop_e\": \"val_e\",\n"
              "           \"prop_f\": \"val_f\"\n"
              "       },\n"
              "       {\n"
              "           \"prop_g\": \"val_g\",\n"
              "           \"prop_h\": \"val_h\"\n"
              "       }\n"
              "    ]\n"
              "}\n"
              )
    file_2 = ("{\n"
              "    \"section_B\": [\n"
              "       {\n"
              "           \"prop_i\": \"val_i\",\n"
              "           \"prop_j\": \"val_j\"\n"
              "       }\n"
              "    ],\n"
              "    \"section_C\": [\n"
              "       {\n"
              "           \"prop_k\": \"val_k\",\n"
              "           \"prop_l\": \"val_l\"\n"
              "       }\n"
              "    ]\n"
              "}\n"
              )

    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=file_1)):
        parser = DescriptionParser()
        description = parser.read("mock_1.pycfg")

    with patch(builtins.__name__ + ".open",
               mock_open(
                   read_data=file_2)):
        parser = DescriptionParser()
        parser.read("mock_2.pycfg",
                    description)

    return description


def testAllSectionsPresent(pyconfig):
    assert('section_A' in pyconfig.data and
           'section_B' in pyconfig.data and
           'section_C' in pyconfig.data)
    assert(len(pyconfig.data.keys()) == 3)


def testSectionA(pyconfig):
    assert len(pyconfig.data['section_A']) == 2
    assert pyconfig.data['section_A'][0] == {
        'prop_a': 'val_a',
        'prop_b': 'val_b'}
    assert pyconfig.data['section_A'][1] == {
        'prop_c': 'val_c',
        'prop_d': 'val_d'}


def testSectionB(pyconfig):
    assert len(pyconfig.data['section_B']) == 3
    assert pyconfig.data['section_B'][0] == {
        'prop_e': 'val_e',
        'prop_f': 'val_f'}
    assert pyconfig.data['section_B'][1] == {
        'prop_g': 'val_g',
        'prop_h': 'val_h'}
    assert pyconfig.data['section_B'][2] == {
        'prop_i': 'val_i',
        'prop_j': 'val_j'}


def testSectionC(pyconfig):
    assert len(pyconfig.data['section_C']) == 1
    assert pyconfig.data['section_C'][0] == {
        'prop_k': 'val_k',
        'prop_l': 'val_l'}
