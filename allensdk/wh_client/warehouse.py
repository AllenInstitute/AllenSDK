# Copyright 2015 Allen Institute for Brain Science
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
        '''Set the internal RMA and well known file download endpoint urls
        based on a warehouse server endpoint.
        
        Parameters
        ----------
        warehouse_base_url_string : string
            url of the warehouse to point to
        '''
        self.warehouse_url = warehouse_base_url_string
        self.well_known_file_endpoint = warehouse_base_url_string + '/api/v2/well_known_file_download/'
        self.rma_endpoint = warehouse_base_url_string + '/api/v2/data'  
    
    
    def set_default_working_directory(self, working_directory):
        '''Set the working directory where files will be saved.
        
        Parameters
        ----------
        working_directory : string
             the absolute path string of the working directory.
        '''
        self.default_working_directory = working_directory
    
    
    def do_rma_query(self, rma_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an RMA query url construction function
        with a corresponding response json traversal function.
        
        Parameters
        ----------
        rma_builder_fn : function
            A function that takes parameters and returns an rma url.
        json_traversal_fn : function
            A function that takes a json-parsed python data structure and returns data from it.
        args : arguments
            Arguments to be passed to the rma builder function.
        kwargs : keyword arguments
            Keyword arguments to be passed to the rma builder function.
        
        Returns
        -------
        any type
            The data extracted from the json response.
        '''
        rma_url = rma_builder_fn(*args, **kwargs) 
                           
        json_parsed_data = self.retrieve_parsed_json_over_http(rma_url)
        
        return json_traversal_fn(json_parsed_data)
    
    
    def get_sample_well_known_file_ids(self, structure_names=['DG']):
        '''Query the current RMA endpoint with a list of structure names
        to get the corresponding well known file ids.
        
        Returns
        -------
        list
            A list of well known file id strings.
        '''
        rma_builder_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).build_rma_url_microarray_data_set_well_known_files(x)
        json_traversal_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).read_json_sample_microarray_slides_well_known_file_id(x)
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, structure_names) 


    def get_cell_types_well_known_file_ids(self, cell_type_names=['DG']):
        '''Query the current RMA endpoint with a list of cell type names to get the corresponding well known file ids for the .hoc files.
        
        Returns
        -------
        list of strings
            A list of well known file id strings.
        '''
        rma_builder_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).build_rma_biophysical_model_well_known_files(x)
        json_traversal_fn = lambda x: RmaCannedQueryCookbook(self.rma_endpoint).read_json_biophysical_model_well_known_file_id(x)
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, cell_type_names) 
    
    
    def cache_cell_types_data(self, cell_type_names, suffix='.hoc', prefix='', working_directory=None,):
        '''Take a list of cell-type names, query the Warehouse RMA to get well-known-files
        download the files, and store them in the working directory.
        
        Parameters
        ----------
        cell_type_names : list of string
            Cell type names to be found in the cell types table in the warehouse
        suffix : string
            Appended to the save file name
        prefix : string
            Prepended to the save file name
        working_directory : string
            Absolute path name where the downloaded well-known files will be stored.
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
        
        Returns
        -------
        dict
            the parsed json schema message
        '''
        schema_url = self.rma_endpoint + '/enumerate.json'
        json_parsed_schema_data = self.retrieve_parsed_json_over_http(schema_url)
        
        return json_parsed_schema_data
    
    
    def construct_well_known_file_download_url(self, well_known_file_id):
        '''Join data warehouse endpoint and id.
        
        Parameters
        ----------
        well_known_file_id : integer or string representing an integer
            well known file id
        
        Returns
        -------
        string
            the well-known-file download url for the current warehouse api server
        '''
        return self.well_known_file_endpoint + str(well_known_file_id)
    
    
    def retrieve_file_over_http(self, url, file_path):
        '''Get a file from the data warehouse and save it.
        
        Parameters
        ----------
        url : string
            Url from which to get the file.
        file_path : string
            Absolute path including the file name to save.
        '''
        try:
            with open(file_path, 'wb') as f:
                response = urllib2.urlopen(url)
                f.write(response.read())
        except urllib2.HTTPError:
            self.log.error("Couldn't retrieve file from %s" % url)
            raise
    
    
    def retrieve_parsed_json_over_http(self, rma_url):
        '''Get the document and put it in a Python data structure
        
        Parameters
        ----------
        rma_url : string
            Full RMA query url.
        
        Returns
        -------
        dict
            Result document as parsed by the JSON library.
        '''
        response = urllib2.urlopen(rma_url)
        json_parsed_data = load(response)
        
        return json_parsed_data
