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
import logging


class Api(object):
    _log = logging.getLogger(__name__)
    default_api_url = 'http://api.brain-map.org'
    
    def __init__(self, api_base_url_string=None):
        if api_base_url_string==None:
            api_base_url_string=Api.default_api_url
            
        self.set_api_urls(api_base_url_string)
        self.default_working_directory = None
    
    
    def set_api_urls(self, api_base_url_string):
        '''Set the internal RMA and well known file download endpoint urls
        based on a api server endpoint.
        
        Parameters
        ----------
        api_base_url_string : string
            url of the api to point to
        '''
        self.api_url = api_base_url_string
        self.well_known_file_endpoint = api_base_url_string + '/api/v2/well_known_file_download/'
        self.rma_endpoint = api_base_url_string + '/api/v2/data'  
    
    
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
        
        Examples
        --------
        `A simple Api subclass example
        <data_api_client.html#creating-new-api-query-classes>`_.
        '''
        rma_url = rma_builder_fn(*args, **kwargs) 

        quoted_rma_url = urllib2.quote(rma_url,';/?:@&=+$,')
                           
        json_parsed_data = self.retrieve_parsed_json_over_http(quoted_rma_url)
        
        return json_traversal_fn(json_parsed_data)
    
    
    def load_api_schema(self):
        '''Download the RMA schema from the current RMA endpoint
        
        Returns
        -------
        dict
            the parsed json schema message
        
        Notes
        -----
        This information and other 
        `Allen Brain Atlas Data Portal Data Model <http://help.brain-map.org/display/api/Data+Model>`_
        documentation is also available as a
        `Class Hierarchy <http://api.brain-map.org/class_hierarchy>`_
        and `Class List <http://api.brain-map.org/class_hierarchy>`_.
        
        '''
        schema_url = self.rma_endpoint + '/enumerate.json'
        json_parsed_schema_data = self.retrieve_parsed_json_over_http(schema_url)
        
        return json_parsed_schema_data
    
    
    def construct_well_known_file_download_url(self, well_known_file_id):
        '''Join data api endpoint and id.
        
        Parameters
        ----------
        well_known_file_id : integer or string representing an integer
            well known file id
        
        Returns
        -------
        string
            the well-known-file download url for the current api api server
        
        See Also
        --------
        retrieve_file_over_http: Can be used to retrieve the file from the url.
        '''
        return self.well_known_file_endpoint + str(well_known_file_id)
    
    
    def retrieve_file_over_http(self, url, file_path):
        '''Get a file from the data api and save it.
        
        Parameters
        ----------
        url : string
            Url[1]_ from which to get the file.
        file_path : string
            Absolute path including the file name to save.
        
        See Also
        --------
        construct_well_known_file_download_url: Can be used to construct the url.
        
        References
        ----------
        .. [1] Allen Brain Atlas Data Portal: `Downloading a WellKnownFile <http://help.brain-map.org/display/api/Downloading+a+WellKnownFile>`_.
        '''
        try:
            with open(file_path, 'wb') as f:
                response = urllib2.urlopen(url)
                f.write(response.read())
        except urllib2.HTTPError:
            self._log.error("Couldn't retrieve file from %s" % url)
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
