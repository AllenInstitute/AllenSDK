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
from mock import patch, mock_open
from allensdk.config.model.description_parser import DescriptionParser
try:
    import __builtin__ as builtins
except:
    import builtins


@pytest.fixture
def multiconfig():
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

    parser = DescriptionParser()

    with patch(builtins.__name__ + ".open",
               mock_open(read_data=file_1)):
        description = parser.read("mock_1.json")

    with patch(builtins.__name__ + ".open",
               mock_open(read_data=file_2)):
        parser.read("mock_2.json", description)

    return description


def testAllSectionsPresent(multiconfig):
    assert ('section_A' in multiconfig.data and
            'section_B' in multiconfig.data and
            'section_C' in multiconfig.data)
    assert len(multiconfig.data.keys()) == 3


def testSectionA(multiconfig):
    assert len(multiconfig.data['section_A']) == 2
    assert multiconfig.data['section_A'][0] == {
        'prop_a': 'val_a',
        'prop_b': 'val_b'}
    assert multiconfig.data['section_A'][1] == {
        'prop_c': 'val_c',
        'prop_d': 'val_d'}


def testSectionB(multiconfig):
    assert len(multiconfig.data['section_B']) == 3
    assert multiconfig.data['section_B'][0] == {
        'prop_e': 'val_e',
        'prop_f': 'val_f'}
    assert multiconfig.data['section_B'][1] == {
        'prop_g': 'val_g',
        'prop_h': 'val_h'}
    assert multiconfig.data['section_B'][2] == {
        'prop_i': 'val_i',
        'prop_j': 'val_j'}


def testSectionC(multiconfig):
    assert len(multiconfig.data['section_C']) == 1
    assert multiconfig.data['section_C'][0] == {
        'prop_k': 'val_k',
        'prop_l': 'val_l'}
