from allensdk.api.queries.ontologies_api import OntologiesApi

import unittest
from mock import MagicMock

class OntologiesApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OntologiesApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.oa = OntologiesApi()
    
    
    def tearDown(self):
        self.oa = None
    
    
    def test_get_structure_graph(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,[graph_id$in1],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        structure_graph_id = 1
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_graph_id)
        
        self.oa.json_msg_query.assert_called_once_with(expected)
    
    
    def test_list_structure_graphs(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::StructureGraph,rma::options[num_rows$eq'all'][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.oa.get_structure_graphs()
        
        self.oa.json_msg_query.assert_called_once_with(expected)
        
    
    def test_list_structure_sets(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::StructureSet,rma::options[num_rows$eq'all'][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.oa.get_structure_sets()
        
        self.oa.json_msg_query.assert_called_once_with(expected)


    def test_list_atlases(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Atlas,rma::options[num_rows$eq'all'][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
               
        self.oa.get_atlases()
        
        self.oa.json_msg_query.assert_called_once_with(expected)

        
    def test_structure_graph_by_name(self):
        expected = u"http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,graph[structure_graphs.name$in'Mouse Brain Atlas'],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_graph_names="'Mouse Brain Atlas'")
        
        self.oa.json_msg_query.assert_called_once_with(expected)


    def test_structure_graphs_by_names(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,graph[structure_graphs.name$in'Mouse Brain Atlas','Human Brain Atlas'],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_graph_names=["'Mouse Brain Atlas'",
                                                      "'Human Brain Atlas'"])
        
        self.oa.json_msg_query.assert_called_once_with(expected)

    
    def test_structure_set_by_id(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,[structure_set_id$in8],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_set_ids=8)
        
        self.oa.json_msg_query.assert_called_once_with(expected)
        
        
    def test_structure_sets_by_ids(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,[structure_set_id$in7,8],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_set_ids=[7,8])
        
        self.oa.json_msg_query.assert_called_once_with(expected)
        
        
    def test_structure_set_by_name(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,structure_sets[name$in'Mouse Connectivity - Summary'],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_set_names=self.oa.quote_string("Mouse Connectivity - Summary"))
        
        self.oa.json_msg_query.assert_called_once_with(expected)


    def test_structure_set_by_names(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,structure_sets[name$in'NHP - Coarse','Mouse Connectivity - Summary'],rma::options[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(structure_set_names=[self.oa.quote_string("NHP - Coarse"),
                                                    self.oa.quote_string("Mouse Connectivity - Summary")])
        
        self.oa.json_msg_query.assert_called_once_with(expected)


    def test_structure_set_no_order(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::criteria,[graph_id$in1],rma::options[num_rows$eq'all'][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_structures(1, order=None)
        
        self.oa.json_msg_query.assert_called_once_with(expected)


    def test_atlas_1(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Atlas,rma::criteria,[id$in1],structure_graph(ontology),graphic_group_labels,rma::include,structure_graph(ontology),graphic_group_labels,rma::options[only$eq'atlases.id,atlases.name,atlases.image_type,ontologies.id,ontologies.name,structure_graphs.id,structure_graphs.name,graphic_group_labels.id,graphic_group_labels.name'][num_rows$eq'all'][count$eqfalse]"
        
        atlas_id = 1
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_atlases_table(atlas_id)
        
        self.oa.json_msg_query.assert_called_once_with(expected)
    
    
    def test_atlas_verbose(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Atlas,rma::criteria,structure_graph(ontology),graphic_group_labels,rma::include,structure_graph(ontology),graphic_group_labels,rma::options[num_rows$eq'all'][count$eqfalse]"
        
        self.oa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.oa.get_atlases_table(brief=False)
        
        self.oa.json_msg_query.assert_called_once_with(expected)

