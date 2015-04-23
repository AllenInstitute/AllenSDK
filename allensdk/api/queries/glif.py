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

from allensdk.api.api import Api

import os, json, logging

class GlifApi(Api):
    def __init__(self, base_uri=None):
        super(GlifApi, self).__init__(base_uri)

        self.metadata = None
        

    def get_neuronal_model(self, neuronal_model_id):
        '''Query the current RMA endpoint with a neuronal_model id
        to get the corresponding well known files and meta data.
        
        Returns
        -------
        dict
            A dictionary containing 
        '''
        self.metadata = self.do_rma_query(self.build_rma_url, self.read_json, neuronal_model_id)
        return self.metadata


    def assert_model_exists(self):
        ''' Make sure that a neuronal model has been downloaded. '''

        if self.metadata is None:
            raise Exception("Neuronal model metadata required.  Please call get_neuronal_model(id)")


    def get_sweeps(self):
        ''' Retrieve ephys sweep information out of downloaded metadata for a neuronal model 
        
        Returns
        -------
        list
            A list of sweeps metadata dictionaries
        '''
        self.assert_model_exists()

        return self.metadata['ephys_sweeps']


    def get_neuron_config(self, output_file_name=None):
        ''' Retrieve a model configuration file from the API, optionally save it to disk, and 
        return the contents of that file as a dictionary.

        Parameters
        ----------
        output_file_name: string
            File name to store the neuron configuration (optional).
        '''

        self.assert_model_exists()

        neuron_config = self.retrieve_parsed_json_over_http(self.metadata['neuron_config_url'])

        if output_file_name:
            with open(output_file_name, 'wb') as f:
                f.write(json.dumps(neuron_config, indent=2))
        
        return neuron_config


    def cache_stimulus_file(self, output_file_name):
        self.assert_model_exists()

        self.retrieve_file_over_http(self.metadata['stimulus_url'], output_file_name)


    def build_rma_url(self, neuronal_model_id, fmt='json'):
        '''Construct a query to find all files related to a GLIF neuronal model.
        
        Parameters
        ----------
        neuronal_model_id : integer or string representation
            key of experiment to retrieve.
            
        Returns
        -------
        string
            RMA query url.
        '''
        
        include_associations = ''.join([
                'neuronal_model_template(well_known_files(well_known_file_type)),',
                'specimen(ephys_sweeps,ephys_result(well_known_files(well_known_file_type))),',
                'well_known_files(well_known_file_type)'])
                
        criteria_associations = ''.join([
                ("[id$eq%d]," % (neuronal_model_id)),
                include_associations])
        
        return ''.join([self.rma_endpoint, 
                        '/query.',
                        fmt,
                        '?q=',
                        'model::NeuronalModel,',
                        'rma::criteria,',
                        criteria_associations,
                        ',rma::include,',
                        include_associations])


    def read_json(self, json_parsed_data):
        ''' Reformat the RMA query results into a more usable dictionary
        
        Parameters
        ----------
        json_parsed_data: dict
            the json response from the Allen Institute API RMA.

        Returns
        -------
        hash
            a dictionary containing fields necessary to run a GLIF model
        '''

        data = {
            'ephys_sweeps': None,
            'neuron_config_url': None,
            'stimulus_url': None            
            }

        neuronal_model = json_parsed_data['msg'][0]

        # sweeps come from the specimen
        try:
            specimen = neuronal_model['specimen']
            ephys_sweeps = specimen['ephys_sweeps']
        except Exception, e:
            logging.warning("Could not find ephys_sweeps for this model")
            ephys_sweeps = None
        
        # neuron config file comes from the neuronal model's well known files
        neuron_config_url = None
        try:
            for wkf in neuronal_model['well_known_files']:
                if wkf['path'].endswith('neuron_config.json'):
                    neuron_config_url = wkf['download_link']
                    break
        except Exception, e:
            logging.warning("Could not find neuron config well_known_file for this model")
            neuron_config_url = None

        # NWB file comes from the ephys_result's well known files
        stimulus_url = None
        try:
            ephys_result = specimen['ephys_result']
            for wkf in ephys_result['well_known_files']:
                if wkf['well_known_file_type'] == 'NWB':
                    stimulus_url = wkf['download_link']
                    break
        except Exception, e:
            logging.warning("Could not find stimulus well_known_file for this model")
            stimulus_url = None
        
        data['ephys_sweeps'] = ephys_sweeps
        data['neuron_config_url'] = neuron_config_url
        data['stimulus_url'] = stimulus_url
        data['neuronal_model'] = neuronal_model
        
        return data



    
        
