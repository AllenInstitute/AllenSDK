import urllib2
from json import load
import os
import logging
from allensdk.wh_client.rma_canned_query_cookbook import RmaCannedQueryCookbook

class Warehouse:
    log = logging.getLogger(__name__)
    default_warehouse_url = 'http://api.brain-map.org'
    
    def __init__(self, warehouse_base_url_string=default_warehouse_url):
        self.set_warehouse_urls(warehouse_base_url_string)
        self.default_working_directory = None
    
    
    def set_warehouse_urls(self, warehouse_base_url_string):
        '''Set the internal RMA and well known file download endpoint urls based on a warehouse server url.
        
        :parameter warehouse_base_url_string: url of the warehouse to point to
        :type warehouse_base_url_string: string
        '''
        self.warehouse_url = warehouse_base_url_string
        self.well_known_file_endpoint = warehouse_base_url_string + '/api/v2/well_known_file_download/'
        self.rma_endpoint = warehouse_base_url_string + '/api/v2/data'  
    
    
    def set_default_working_directory(self, working_directory):
        '''Set the working directory where files will be saved.
        
        :parameter working_directory: the absolute path string of the working directory.
        :type working_directory: string
        '''
        self.default_working_directory = working_directory
    
    
    def do_rma_query(self, rma_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an RMA query url construction function with a corresponding response json traversal function.
        
        :parameter rma_builder_fn: a function that takes parameters and returns an rma url
        :type rma_builder_fn: function
        :parameter json_traversal_fn: a function that takes a json-parsed python data structure and returns data from it.
        :type json_traversal_fn: function
        :parameter args: arguments to be passed to the rma builder function
        :type args: arguments
        :parameter kwargs: keyword arguments to be passed to the rma builder function
        :type kwargs: keyword arguments
        :returns: the data extracted from the json response.
        :rtype: arbitrary
        '''
        rma_url = rma_builder_fn(*args, **kwargs) 
                           
        json_parsed_data = self.retrieve_parsed_json_over_http(rma_url)
        
        return json_traversal_fn(json_parsed_data)
    
    
    def get_sample_well_known_file_ids(self, structure_names=['DG']):
        '''Query the current RMA endpoint with a list of structure names to get the corresponding well known file ids.
        
        :returns: a list of well known file id strings
        :rtype: list
        '''
        rma_builder_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).build_rma_url_microarray_data_set_well_known_files(x)
        json_traversal_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).read_json_sample_microarray_slides_well_known_file_id(x)
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, structure_names) 


    def get_cell_types_well_known_file_ids(self, cell_type_names=['DG']):
        '''Query the current RMA endpoint with a list of cell type names to get the corresponding well known file ids for the .hoc files.
        
        :returns: a list of well known file id strings
        :rtype: list
        '''
        rma_builder_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).build_rma_biophysical_model_well_known_files(x)
        json_traversal_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).read_json_biophysical_model_well_known_file_id(x)
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, cell_type_names) 
    
    
    def cache_cell_types_data(self, cell_type_names, suffix='.hoc', prefix='', working_directory=None,):
        '''Take a list of cell-type names, query the Warehouse RMA to get well-known-files, download the files, and store them in the working directory.
            
        :parameter cell_type_names: cell type names to be found in the cell types table in the warehouse
        :type cell_type_names: list
        :parameter suffix: appended to the save file name
        :type suffix: string
        :parameter prefix: prepended to the save file name
        :type prefix: string
        :parameter working_directory: absolute path name where the downloaded well-known files will be stored.
        :type working_directory: string
        '''
        if working_directory == None:
            working_directory = self.default_working_directory
        
        well_known_file_id_dict = self.get_cell_types_well_known_file_ids(cell_type_names)
        
        for cell_type_name, well_known_file_id in well_known_file_id_dict.items():
            well_known_file_url = self.construct_well_known_file_download_url(well_known_file_id)
            cached_file_path = os.path.join(working_directory, "%s%s%s" % (prefix, cell_type_name, suffix))
            self.retrieve_file_over_http(well_known_file_url, cached_file_path)
    
    
    def load_warehouse_schema(self):
        '''Download the RMA schema from the current RMA endpoint
        
        :returns: the parsed json schema message
        :rtype: hash
        '''
        schema_url = self.rma_endpoint + '/enumerate.json'
        json_parsed_schema_data = self.retrieve_parsed_json_over_http(schema_url)
        
        return json_parsed_schema_data
    
    
    def construct_well_known_file_download_url(self, well_known_file_id):
        ''' 
        :parameter well_known_file_id: a well known file id
        :type well_known_file_id: integer or string representing an integer
        :returns: the well-known-file download url for the current warehouse api server
        :rtype: string
        '''
        return self.well_known_file_endpoint + str(well_known_file_id)
    
    
    def retrieve_file_over_http(self, url, file_path):
        '''Very simple method to get a file via http and save it.
                
        :parameter url: the url from which to get the file
        :type url: string
        :parameter file_path: the absolute path including the file name of the file to be saved
        :type file_path: string
        '''
        try:
            with open(file_path, 'wb') as f:
                response = urllib2.urlopen(url)
                f.write(response.read())
        except urllib2.HTTPError:
            self.log.error("Couldn't retrieve file from %s" % url)
            raise
    
    
    def retrieve_parsed_json_over_http(self, rma_url):
        '''Very simple method to get a json message via http and parse it into a Python data structure
        
        :parameter rma_url: the url
        :type rma_url: string
        :returns: the response as parsed by the JSON library.
        :rtype: hash
        '''
        response = urllib2.urlopen(rma_url)
        json_parsed_data = load(response)
        
        return json_parsed_data
