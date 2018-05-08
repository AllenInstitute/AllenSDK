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
from allensdk.api.queries.annotated_section_data_sets_api import \
    AnnotatedSectionDataSetsApi
import pytest
from mock import MagicMock


@pytest.fixture
def annotated():
    asdsa = AnnotatedSectionDataSetsApi()

    asdsa.json_msg_query = MagicMock(name='json_msg_query')

    return asdsa


def test_get_annotated_section_data_set(annotated):
    annotated.get_annotated_section_data_sets(
        structures=[112763676],
        intensity_values=["High", "Low", "Medium"],
        density_values=["High", "Low"],
        pattern_values=["Full"],
        age_names=["E11.5", "13.5"])

    annotated.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/annotated_section_data_sets.json"
        "?structures=112763676&intensity_values='High','Low','Medium'"
        "&density_values='High','Low'"
        "&pattern_values='Full'&age_names='E11.5','13.5'")


def test_get_compound_annotated_section_data_set(annotated):
    annotated.get_annotated_section_data_sets(
        structures=[112763676],
        intensity_values=["High", "Low", "Medium"],
        density_values=["High", "Low"],
        pattern_values=["Full"],
        age_names=["E11.5", "13.5"])

    annotated.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/annotated_section_data_sets.json?"
        "structures=112763676"
        "&intensity_values='High','Low','Medium'&density_values='High','Low'"
        "&pattern_values='Full'"
        "&age_names='E11.5','13.5'")


def test_get_annotated_section_data_set_via_rma(annotated):
    annotated.json_msg_query = \
        MagicMock(name='json_msg_query')

    annotated.get_compound_annotated_section_data_sets(
        [{'structures': [112763676],
          'intensity_values': ['High', 'Low'],
          'link': 'or'},
         {'structures': [112763686],
          'intensity_values': ['Low']}])

    annotated.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org"
        "/api/v2/compound_annotated_section_data_sets.json"
        "?query=[structures $in 112763676 : intensity_values $in 'High','Low']"
        " or [structures $in 112763686 : intensity_values $in 'Low']")
