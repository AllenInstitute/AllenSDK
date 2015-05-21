from allensdk.api.queries.svg.svg_api import SvgApi

import unittest, json
from mock import patch, mock_open

class SvgApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SvgApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        pass
    
    
    def tearDown(self):
        pass
    
    
    def test_structure_graph_1(self):
        pass
    
if '__main__' == __name__:
    svg = SvgApi()
    print(svg.get_svg(100960033, [28]))
