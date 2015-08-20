from allensdk.api.queries.annotated_section_data_sets_api import \
    AnnotatedSectionDataSetsApi

import unittest, json
from mock import MagicMock

class AnnotatedSectionDataSetsApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AnnotatedSectionDataSetsApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.asdsa = AnnotatedSectionDataSetsApi()
    
    
    def tearDown(self):
        self.asdsa = None


    def test_get_annotated_section_data_set(self):
        expected = "http://api.brain-map.org/api/v2/annotated_section_data_sets.json?structures=112763676&intensity_values='High','Low','Medium'&density_values='High','Low'&pattern_values='Full'&age_names='E11.5','13.5'"
        
        self.asdsa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.asdsa.get_annotated_section_data_sets(
            structures=[112763676],
            intensity_values=["High", "Low", "Medium"],
            density_values=["High","Low"],
            pattern_values=["Full"],
            age_names=["E11.5", "13.5"])
        
        self.asdsa.json_msg_query.assert_called_once_with(expected)


    def test_get_compound_annotated_section_data_set(self):
        expected = "http://api.brain-map.org/api/v2/annotated_section_data_sets.json?structures=112763676&intensity_values='High','Low','Medium'&density_values='High','Low'&pattern_values='Full'&age_names='E11.5','13.5'"
        
        self.asdsa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.asdsa.get_annotated_section_data_sets(
            structures=[112763676],
            intensity_values=["High", "Low", "Medium"],
            density_values=["High","Low"],
            pattern_values=["Full"],
            age_names=["E11.5", "13.5"])
        
        self.asdsa.json_msg_query.assert_called_once_with(expected)
    

    def test_get_annotated_section_data_set_via_rma(self):
        expected = "http://api.brain-map.org/api/v2/compound_annotated_section_data_sets.json?query=[structures $in 112763676 : intensity_values $in 'High','Low'] or [structures $in 112763686 : intensity_values $in 'Low']"
        
        self.asdsa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.asdsa.get_compound_annotated_section_data_sets(
            [{'structures' : [112763676],
              'intensity_values' : ['High', 'Low'],
              'link' : 'or' },
             {'structures' : [112763686],
              'intensity_values' : ['Low']}])
        
        self.asdsa.json_msg_query.assert_called_once_with(expected)