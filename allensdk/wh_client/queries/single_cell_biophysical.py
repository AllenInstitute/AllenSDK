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

class SingleCellBiophysical(Warehouse):
    def __init__(self, base_uri=None):
        super(SingleCellBiophysical, self).__init__(base_uri)
        self.cache_stimulus = False
    
    
    def build_rma_url_biophysical_neuronal_model_run(self, neuronal_model_run_id, fmt='json'):
        '''Construct a query to find all files related to a neuronal model run.
        
        Parameters
        ----------
        neuronal_model_run_id : integer or string representation
            key of experiment to retrieve.
            
        Returns
        -------
        string
            RMA query url.
        '''
        include_associations = ''.join([
            'neuronal_model',
            '(neuronal_model_template(well_known_files),',
            'specimen(neuron_reconstructions(well_known_files),',
            'ephys_sweeps),',
            'well_known_files),',
            'well_known_files'])
        criteria_associations = ''.join([
            ("[id$eq%d]," % (neuronal_model_run_id)),
            include_associations])
        
        return ''.join([self.rma_endpoint, 
                        '/query.',
                        fmt,
                        '?q=',
                        'model::NeuronalModelRun,',
                        'rma::criteria,',
                        criteria_associations,
                        ',rma::include,',
                        include_associations])
    
    
    def read_json(self, json_parsed_data):
        '''Get the list of well_known_file ids from a response body containing nested sample,microarray_slides,well_known_files.
        
        :parameter json_parsed_data: the json response from the Allen Institute Warehouse RMA.
        :type json_parsed_data: hash
        :returns: a list of well_known_file ids
        :rtype: list of strings
        '''
        ids = {}
        
        if 'msg' in json_parsed_data:
            for neuronal_model_run in json_parsed_data['msg']:
                if self.cache_stimulus == True and 'well_known_files' in neuronal_model_run:
                    for well_known_file in neuronal_model_run['well_known_files']:
                        if 'id' in well_known_file and 'path' in well_known_file:
                            ids[str(well_known_file['id'])] = \
                                os.path.split(well_known_file['path'])[1]

                if 'neuronal_model' in neuronal_model_run:
                    neuronal_model = neuronal_model_run['neuronal_model']
                    
                    if 'well_known_files' in neuronal_model:
                        for well_known_file in neuronal_model['well_known_files']:
                            if 'id' in well_known_file and 'path' in well_known_file:
                                ids[str(well_known_file['id'])] = \
                                    os.path.split(well_known_file['path'])[1]
                    
                    if 'neuronal_model_template' in neuronal_model:
                        neuronal_model_template = neuronal_model['neuronal_model_template']
                        if 'well_known_files' in neuronal_model_template:
                            for well_known_file in neuronal_model_template['well_known_files']:
                                if 'id' in well_known_file and 'path' in well_known_file:
                                    ids[str(well_known_file['id'])] = \
                                        os.path.split(well_known_file['path'])[1]
                    
                    if 'specimen' in neuronal_model:
                        specimen = neuronal_model['specimen']
                        if 'neuron_reconstructions' in specimen:
                            for neuron_reconstruction in specimen['neuron_reconstructions']:
                                if 'well_known_files' in neuron_reconstruction:
                                    for well_known_file in neuron_reconstruction['well_known_files']:
                                        if 'id' in well_known_file and 'path' in well_known_file:
                                            ids[str(well_known_file['id'])] = \
                                                os.path.split(well_known_file['path'])[1]
        
        return ids
    
    
    def get_well_known_file_ids(self, neuronal_model_run_id):
        '''Query the current RMA endpoint with a neuronal_model_run id
        to get the corresponding well known file ids.
        
        Returns
        -------
        list
            A list of well known file id strings.
        '''
        rma_builder_fn = self.build_rma_url_biophysical_neuronal_model_run
        json_traversal_fn = self.read_json
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, neuronal_model_run_id)
    
    
    def cache_data(self,
                   neuronal_model_run_id,
                   working_directory=None):
        '''Take a an experiment id, query the Warehouse RMA to get well-known-files
        download the files, and store them in the working directory.
        
        Parameters
        ----------
        neuronal_model_run_id : int or string representation
            found in the neuronal_model_run table in the warehouse
        working_directory : string
            Absolute path name where the downloaded well-known files will be stored.
        '''
        if working_directory == None:
            working_directory = self.default_working_directory
        
        well_known_file_id_dict = self.get_well_known_file_ids(neuronal_model_run_id)
        
        for well_known_id, filename in well_known_file_id_dict.items():
            well_known_file_url = self.construct_well_known_file_download_url(well_known_id)
            cached_file_path = os.path.join(working_directory, filename)
            self.retrieve_file_over_http(well_known_file_url, cached_file_path)
