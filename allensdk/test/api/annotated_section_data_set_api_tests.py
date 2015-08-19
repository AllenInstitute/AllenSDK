from allensdk.api.queries.annotated_section_data_sets_api import \
    AnnotatedSectionDataSetsApi

import unittest, json
from mock import patch, mock_open

class AnnotatedSectionDataSetsApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AnnotatedSectionDataSetsApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        pass
    
    
    def tearDown(self):
        pass
    
    
    def test_structure_graph_1(self):
        pass
    
if '__main__' == __name__:
    asdsa = AnnotatedSectionDataSetsApi()
    print(json.dumps(asdsa.get_annotated_section_data_sets_via_rma(
        structures=[112763676],
        intensity_values=["High", "Low", "Medium"],
        density_values=["High","Low"],
        pattern_values=["Full"],
        age_names=["E11.5", "13.5"]), indent=2))
    
    #print(json.dumps(asdsa.get_compound_annotated_section_data_sets(
    #    [{'structures' : [112763676],
    #      'intensity_values' : ['High', 'Low'],
    #      'link' : 'or' },
    #     {'structures' : [112763686],
    #      'intensity_values' : ['Low']}]), indent=2))