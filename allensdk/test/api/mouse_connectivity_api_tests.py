from allensdk.api.queries.annotated_section_data_sets.annotated_section_data_sets_api import \
    AnnotatedSectionDataSetsApi

import unittest, json
from mock import patch, mock_open
from allensdk.api.queries.connectivity.mouse_connectivity_api import MouseConnectivityApi

class MouseConnectivityApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MouseConnectivityApiTests, self).__init__(*args, **kwargs)
        self.mca = None
    
    
    def setUp(self):
        self.mca = MouseConnectivityApi()
    
    
    def tearDown(self):
        self.mca = None
    
    
    def test_api_doc_url_all_experiments(self):
        '''
        Notes
        -----
        Expected link is slightly modified for URL escaping, ?q= and json format.
        
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'All experiments in the "Mouse Connectivity Projection" Product'.
        '''
        expected = 'http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,products[id$eq5],rma::include,specimen(stereotaxic_injections(primary_injection_structure,structures))'
        
        actual = self.mca.build_query()
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_injection_in_visp(self):
        '''
        Notes
        -----
        Expected link is slightly modified for URL escaping, ?q= and json format.
        
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'All experiments with an injection in the primary visual area (VISp, structure_id=385)'.
        '''
        expected = 'http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,products[id$eq5],rma::include,specimen(stereotaxic_injections(primary_injection_structure,structures[id$eq385]))'
        
        visp_structure_id = 385
        actual = self.mca.build_query(visp_structure_id)
        
        self.assertEqual(actual, expected)
        
        
    def test_api_doc_url_detailed_metadata_visp(self):
        '''
        Notes
        -----
        Expected link is slightly modified for quote type, ?q= and json format.
        
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'Detailed metadata for one experiment with injection in the VISp (id=126862385)'.
        '''
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,[id$eq126862385],rma::include,specimen(stereotaxic_injections(primary_injection_structure,structures,stereotaxic_injection_coordinates)),equalization,sub_images,rma::options[order$eq'sub_images.section_number$asc']"
        
        experiment_id = 126862385
        actual = self.mca.build_detail_query(experiment_id)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_projection_image_meta_information(self):
        '''
        Notes
        -----
        Expected link is slightly modified for url escaping, ?q= and json format.
        
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'RMA query to fetch meta-information of one projection image'.
        '''
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,[id$eq126862385],rma::include,equalization,sub_images[section_number$eq74]"
        
        experiment_id = 126862385
        section_number = 74
        actual = self.mca.build_projection_image_meta_info(experiment_id,
                                                           section_number)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_projection_image_downsampled(self):
        '''
        Notes
        -----
        Expected link is slightly modified for url escaping, ?q= and json format.
        
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'RMA query to fetch meta-information of one projection image'.
        '''
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,[id$eq126862385],rma::include,equalization,sub_images[section_number$eq74]"
        
        experiment_id = 126862385
        section_number = 74
        actual = self.mca.build_projection_image_meta_info(experiment_id,
                                                           section_number)
        
        self.assertEqual(actual, expected)