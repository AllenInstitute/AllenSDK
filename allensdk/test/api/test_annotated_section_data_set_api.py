# Copyright 2016 Allen Institute for Brain Science
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
