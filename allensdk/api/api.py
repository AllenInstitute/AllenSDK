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


import requests
from contextlib import closing
import allensdk.core.json_utilities as json_utilities
import pandas as pd
import logging
import os
import errno
import warnings


class Api(object):
    _log = logging.getLogger(__name__)
    default_api_url = 'http://api.brain-map.org'
    download_url = 'http://download.alleninstitute.org'

    def __init__(self, api_base_url_string=None):
        if api_base_url_string is None:
            api_base_url_string = Api.default_api_url

        self.set_api_urls(api_base_url_string)
        self.default_working_directory = os.getcwd()

    def set_api_urls(self, api_base_url_string):
        '''Set the internal RMA and well known file download endpoint urls
        based on a api server endpoint.

        Parameters
        ----------
        api_base_url_string : string
            url of the api to point to
        '''
        self.api_url = api_base_url_string

        # http://help.brain-map.org/display/api/Downloading+a+WellKnownFile
        self.well_known_file_endpoint = api_base_url_string + \
            '/api/v2/well_known_file_download'

        # http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data
        self.grid_data_endpoint = api_base_url_string + '/grid_data'

        # http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG
        self.svg_endpoint = api_base_url_string + '/api/v2/svg'
        self.svg_download_endpoint = api_base_url_string + '/api/v2/svg_download'

        # http://help.brain-map.org/display/api/Downloading+an+Ontology%27s+Structure+Graph
        self.structure_graph_endpoint = api_base_url_string + \
            '/api/v2/structure_graph_download'

        # http://help.brain-map.org/display/api/Searching+a+Specimen+or+Structure+Tree
        self.tree_search_endpoint = api_base_url_string + '/api/v2/tree_search'

        # http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets
        self.annotated_section_data_sets_endpoint = api_base_url_string + \
            '/api/v2/annotated_section_data_sets'
        self.compound_annotated_section_data_sets_endpoint = api_base_url_string + \
            '/api/v2/compound_annotated_section_data_sets'

        # http://help.brain-map.org/display/api/Image-to-Image+Synchronization#Image-to-ImageSynchronization-ImagetoImage
        self.image_to_atlas_endpoint = api_base_url_string + '/api/v2/image_to_atlas'
        self.image_to_image_endpoint = api_base_url_string + '/api/v2/image_to_image'
        self.image_to_image_2d_endpoint = api_base_url_string + '/api/v2/image_to_image_2d'
        self.reference_to_image_endpoint = api_base_url_string + '/api/v2/reference_to_image'
        self.image_to_reference_endpoint = api_base_url_string + '/api/v2/image_to_reference'
        self.structure_to_image_endpoint = api_base_url_string + '/api/v2/structure_to_image'

        # http://help.brain-map.org/display/mouseconnectivity/API
        self.section_image_download_endpoint = api_base_url_string + \
            '/api/v2/section_image_download'
        self.atlas_image_download_endpoint = api_base_url_string + \
            '/api/v2/atlas_image_download'
        self.projection_image_download_endpoint = api_base_url_string + \
            '/api/v2/projection_image_download'
        self.image_download_endpoint = api_base_url_string + \
            '/api/v2/image_download'
        self.informatics_archive_endpoint = Api.download_url + '/informatics-archive'

        self.rma_endpoint = api_base_url_string + '/api/v2/data'

    def set_default_working_directory(self, working_directory):
        '''Set the working directory where files will be saved.

        Parameters
        ----------
        working_directory : string
             the absolute path string of the working directory.
        '''
        self.default_working_directory = working_directory

    def read_data(self, parsed_json):
        '''Return the message data from the parsed query.

        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.

        Notes
        -----
        See `API Response Formats - Response Envelope <http://help.brain-map.org/display/api/API+Response+Formats#APIResponseFormats-ResponseEnvelope>`_
        for additional documentation.
        '''
        return parsed_json['msg']

    def json_msg_query(self, url, dataframe=False):
        ''' Common case where the url is fully constructed
            and the response data is stored in the 'msg' field.

        Parameters
        ----------
        url : string
            Where to get the data in json form
        dataframe : boolean
            True converts to a pandas dataframe, False (default) doesn't

        Returns
        -------
        dict or DataFrame
            returned data; type depends on dataframe option
        '''

        data = self.do_query(lambda *a, **k: url,
                             self.read_data)

        if dataframe is True:
            warnings.warn("dataframe argument is deprecated", DeprecationWarning)
            data = pd.DataFrame(data)

        return data

    def do_query(self, url_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an query url construction function
        with a corresponding response json traversal function.

        Parameters
        ----------
        url_builder_fn : function
            A function that takes parameters and returns an rma url.
        json_traversal_fn : function
            A function that takes a json-parsed python data structure and returns data from it.
        post : boolean, optional kwarg
            True does an HTTP POST, False (default) does a GET
        args : arguments
            Arguments to be passed to the url builder function.
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
        api_url = url_builder_fn(*args, **kwargs)

        post = kwargs.get('post', False)

        json_parsed_data = self.retrieve_parsed_json_over_http(api_url, post)

        return json_traversal_fn(json_parsed_data)

    def do_rma_query(self, rma_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an RMA query url construction function
        with a corresponding response json traversal function.

        ..note:: Deprecated in AllenSDK 0.9.2
            `do_rma_query` will be removed in AllenSDK 1.0, it is replaced by
            `do_query` because the latter is more general.

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
        return self.do_query(rma_builder_fn, json_traversal_fn, *args, **kwargs)

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
        json_parsed_schema_data = self.retrieve_parsed_json_over_http(
            schema_url)

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
        return self.well_known_file_endpoint + '/' + str(well_known_file_id)

    def cleanup_truncated_file(self, file_path):
        '''Helper for removing files.

        Parameters
        ----------
        file_path : string
            Absolute path including the file name to remove.'''
        try:
            os.remove(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

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

        self._log.info("Downloading URL: %s", url)
        
        from requests_toolbelt import exceptions
        from requests_toolbelt.downloadutils import stream

        try:
            with closing(requests.get(url,
                                      stream=True,
                                      timeout=(9.05,31.1))) as response:
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    stream.stream_response_to_file(response, path=f)
        except exceptions.StreamingError as e:
            self._log.error("Couldn't retrieve file %s from %s (streaming)." % (file_path,url))
            self.cleanup_truncated_file(file_path)
            raise
        except requests.exceptions.ConnectionError as e:
            self._log.error("Couldn't retrieve file %s from %s (connection)." % (file_path,url))
            self.cleanup_truncated_file(file_path)
            raise
        except requests.exceptions.ReadTimeout as e:
            self._log.error("Couldn't retrieve file %s from %s (timeout)." % (file_path,url))
            self.cleanup_truncated_file(file_path)
            raise
        except requests.exceptions.RequestException as e:
            self._log.error("Couldn't retrieve file %s from %s (request)." % (file_path,url))
            self.cleanup_truncated_file(file_path)
            raise
        except Exception as e:
            self._log.error("Couldn't retrieve file %s from %s" % (file_path, url))
            self.cleanup_truncated_file(file_path)
            raise


    def retrieve_parsed_json_over_http(self, url, post=False):
        '''Get the document and put it in a Python data structure

        Parameters
        ----------
        url : string
            Full API query url.
        post : boolean
            True does an HTTP POST, False (default) encodes the URL and does a GET

        Returns
        -------
        dict
            Result document as parsed by the JSON library.
        '''
        self._log.info("Downloading URL: %s", url)
        
        if post is False:
            data = json_utilities.read_url_get(
                requests.utils.quote(url,
                                     ';/?:@&=+$,'))
        else:
            data = json_utilities.read_url_post(url)

        return data

    def retrieve_xml_over_http(self, url):
        '''Get the document and put it in a Python data structure

        Parameters
        ----------
        url : string
            Full API query url.

        Returns
        -------
        string
            Unparsed xml string.
        '''
        self._log.info("Downloading URL: %s", url)
                
        response = requests.get(url)

        return response.content
