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

from allensdk.wh_client.warehouse import Warehouse
import os

class LegacyBiophysicalModel(Warehouse):
    def __init__(self, base_uri=None):
        super(LegacyBiophysicalModel, self).__init__(base_uri)
    
    def build_rma_biophysical_model_well_known_files(self, model_names=['DG']):
        '''Build a query relating biophysical models to well known files.
         
        :parameter model_names: the biophysical model names
        :type model_names: list of strings
        :returns: rma query url for the current rma entpoint
        :rtype: string
        '''
        return ''.join([self.rma_endpoint, 
                        '/query.json?q=',
                        'model::BiophysicalModel,',
                        'rma::criteria,',
                        ("[name$in'%s']," % "','".join(model_names)),
                        'rma::include,',
                        'well_known_files[id]'])
    
    
    def read_json_biophysical_model_well_known_file_id(self, json_parsed_data):
        '''Get the list of well_known_file ids from a response body containing nested biophysical_models,well_known_files.
        
        :parameter json_parsed_data: the json response from the Allen Institute Warehouse RMA.
        :type json_parsed_data: hash
        :returns: a dict of biophysical_model name to well-known-file id mappings
        :rtype: dict
        '''
        model_to_file_id_dict = {}
        
        if 'msg' in json_parsed_data:
            for biophysical_model in json_parsed_data['msg']:
                        current_model_name = None
                        
                        if 'name' in biophysical_model:
                            current_model_name = biophysical_model['name']
                        if 'well_known_files' in biophysical_model:
                            for well_known_file in biophysical_model['well_known_files']:
                                if 'id' in well_known_file:
                                    model_to_file_id_dict[current_model_name] = str(well_known_file['id'])
        
        return model_to_file_id_dict
    
    
    def get_cell_types_well_known_file_ids(self, cell_type_names=['DG']):
        '''Query the current RMA endpoint with a list of cell type names to get the corresponding well known file ids for the .hoc files.
        
        Returns
        -------
        list of strings
            A list of well known file id strings.
        '''
        rma_builder_fn = self.build_rma_biophysical_model_well_known_files
        json_traversal_fn = self.read_json_biophysical_model_well_known_file_id
        
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

