from allensdk.api.queries.ontologies_api import OntologiesApi

import unittest, json
from mock import patch, mock_open

class OntologiesApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OntologiesApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.oa = OntologiesApi()
    
    
    def tearDown(self):
        self.oa = None
    
    
    def test_structure_all_graphs(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Atlas,rma::criteria,structure_graph(ontology),graphic_group_labels,rma::include,structure_graph(ontology),graphic_group_labels,rma::options[only$eq'atlases.id,atlases.name,atlases.image_type,ontologies.id,ontologies.name,structure_graphs.id,structure_graphs.name,graphic_group_labels.id,graphic_group_labels.name']"
        
        actual = self.oa.build_atlases_query()
        
        self.assertEqual(expected, actual)
    
    
    def test_atlas_1(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Atlas,rma::criteria,[id$eq1],structure_graph(ontology),graphic_group_labels,rma::include,[id$eq1],structure_graph(ontology),graphic_group_labels,rma::options[only$eq'atlases.id,atlases.name,atlases.image_type,ontologies.id,ontologies.name,structure_graphs.id,structure_graphs.name,graphic_group_labels.id,graphic_group_labels.name']"
        
        atlas_id = 1
        actual = self.oa.build_atlases_query(atlas_id)
        
        self.assertEqual(expected, actual)
    
    
    def test_build_structure_query(self):
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::Structure,rma::include,[graph_id$eq1],rma::options[only$eqstructures.id,structures.parent_structure_id,structures.acronym,structures.graph_order,structures.color_hex_triplet,structures.structure_id_path,structures.name][num_rows$eq'all'][order$eqstructures.graph_order]"
        
        structure_graph_id = 1
        actual = self.oa.build_structure_query(structure_graph_id)
        
        self.assertEqual(expected, actual)
        
        
    def test_structure_graph_1(self):
        expected = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
        
        structure_graph_id = 1
        actual = self.oa.build_structure_graph_query(structure_graph_id)
        
        self.assertEqual(expected, actual)
