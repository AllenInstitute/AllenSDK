from allensdk.api.queries.annotated_section_data_sets_api import \
    AnnotatedSectionDataSetsApi

import pytest, json
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
        density_values=["High","Low"],
        pattern_values=["Full"],
        age_names=["E11.5", "13.5"])

    expected = "http://api.brain-map.org/api/v2/annotated_section_data_sets.json?structures=112763676&intensity_values='High','Low','Medium'&density_values='High','Low'&pattern_values='Full'&age_names='E11.5','13.5'"
    annotated.json_msg_query.assert_called_once_with(expected)


def test_get_compound_annotated_section_data_set(annotated):
    annotated.get_annotated_section_data_sets(
        structures=[112763676],
        intensity_values=["High", "Low", "Medium"],
        density_values=["High","Low"],
        pattern_values=["Full"],
        age_names=["E11.5", "13.5"])
    
    expected = "http://api.brain-map.org/api/v2/annotated_section_data_sets.json?structures=112763676&intensity_values='High','Low','Medium'&density_values='High','Low'&pattern_values='Full'&age_names='E11.5','13.5'"
    annotated.json_msg_query.assert_called_once_with(expected)


def test_get_annotated_section_data_set_via_rma(annotated):
    annotated.json_msg_query = \
        MagicMock(name='json_msg_query')
           
    annotated.get_compound_annotated_section_data_sets(
        [{'structures' : [112763676],
          'intensity_values' : ['High', 'Low'],
          'link' : 'or' },
         {'structures' : [112763686],
          'intensity_values' : ['Low']}])

    expected = "http://api.brain-map.org/api/v2/compound_annotated_section_data_sets.json?query=[structures $in 112763676 : intensity_values $in 'High','Low'] or [structures $in 112763686 : intensity_values $in 'Low']"
    annotated.json_msg_query.assert_called_once_with(expected)