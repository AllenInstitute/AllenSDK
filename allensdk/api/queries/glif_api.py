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

import json, logging

class GlifApi(Api):
    def __init__(self, base_uri=None):
        super(GlifApi, self).__init__(base_uri)

        self.neuronal_model = None
        self.ephys_sweeps = None
        self.stimulus_url = None
        self.neuron_config_url = None

    def list_neuronal_models(self):
        ''' Query the API for a list of all GLIF neuronal models. 

        Returns
        -------
        list
            Meta data for all GLIF neuronal models.
        '''
        
        return self.do_query(self.build_neuronal_model_list_rma_url, self.read_neuronal_model_list_json)


    def build_neuronal_model_list_rma_url(self, fmt='json'):
        include_associations = "specimen(ephys_result),neuronal_model_template[name$il'*LIF*']"

        url = ''.join([self.rma_endpoint, 
                       '/query.',
                       fmt,
                       '?q=',
                       'model::NeuronalModel,',
                       'rma::include,',
                       include_associations,
                       ',rma::options[num_rows$eq2000]'])

        return url
        

    def read_neuronal_model_list_json(self, json_parsed_data):
        ''' Return basic meta data for all neuronal models'''
        return json_parsed_data['msg']


    def get_neuronal_model(self, neuronal_model_id):
        '''Query the current RMA endpoint with a neuronal_model id
        to get the corresponding well known files and meta data.
        
        Returns
        -------
        dict
            A dictionary containing 
        '''
        
        self.metadata = self.do_rma_query(self.build_neuronal_model_rma_url, self.read_neuronal_model_json, neuronal_model_id)
        return self.metadata


    def get_ephys_sweeps(self):
        ''' Retrieve ephys sweep information out of downloaded metadata for a neuronal model 
        
        Returns
        -------
        list
            A list of sweeps metadata dictionaries
        '''
        return self.ephys_sweeps


    def get_neuron_config(self, output_file_name=None):
        ''' Retrieve a model configuration file from the API, optionally save it to disk, and 
        return the contents of that file as a dictionary.

        Parameters
        ----------
        output_file_name: string
            File name to store the neuron configuration (optional).
        '''
        if self.neuron_config_url is None:
            raise Exception("URL for neuron config file is empty.")

        neuron_config = self.retrieve_parsed_json_over_http(self.api_url + self.neuron_config_url)

        if output_file_name:
            with open(output_file_name, 'wb') as f:
                f.write(json.dumps(neuron_config, indent=2))
        
        return neuron_config

    def cache_stimulus_file(self, output_file_name):
        ''' Download the NWB file for the current neuronal model and save it to a file.

        Parameters
        ----------
        output_file_name: string
            File name to store the NWB file.
        '''
        if self.stimulus_url is None:
            raise Exception("URL for stimulus file is empty.")

        self.retrieve_file_over_http(self.api_url + self.metadata['stimulus_url'], output_file_name)


    def build_neuronal_model_rma_url(self, neuronal_model_id, fmt='json'):
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
                ("[id$eq%d]" % (neuronal_model_id))])
        
        url = ''.join([self.rma_endpoint, 
                       '/query.',
                       fmt,
                       '?q=',
                       'model::NeuronalModel,',
                       'rma::criteria,',
                       criteria_associations,
                       ',rma::include,',
                       include_associations])

        return url


    def read_neuronal_model_json(self, json_parsed_data):
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

        self.ephys_sweeps = None
        self.neuron_config_url = None
        self.stimulus_url = None            
        self.neuronal_model = json_parsed_data['msg'][0]

        # sweeps come from the specimen
        try:
            specimen = self.neuronal_model['specimen']
            self.ephys_sweeps = specimen['ephys_sweeps']
        except Exception, e:
            self.ephys_sweeps = None
            
        if self.ephys_sweeps is None:
            logging.warning("Could not find ephys_sweeps for this model (%d)" % self.neuronal_model['id'])

        
        # neuron config file comes from the neuronal model's well known files
        try:
            for wkf in self.neuronal_model['well_known_files']:
                if wkf['path'].endswith('neuron_config.json'):
                    self.neuron_config_url = wkf['download_link']
                    break
        except Exception, e:
            self.neuron_config_url = None

        if self.neuron_config_url is None:
            logging.warning("Could not find neuron config well_known_file for this model (%d)" % self.neuronal_model['id'])

        # NWB file comes from the ephys_result's well known files
        try:
            ephys_result = specimen['ephys_result']
            for wkf in ephys_result['well_known_files']:
                if wkf['well_known_file_type']['name'] == 'NWB':
                    self.stimulus_url = wkf['download_link']
                    break
        except Exception, e:
            self.stimulus_url = None

        if self.stimulus_url is None:
            logging.warning("Could not find stimulus well_known_file for this model (%d)" % self.neuronal_model['id'])


        data = {
            'neuron_config_url': self.neuron_config_url,
            'stimulus_url': self.stimulus_url,
            'ephys_sweeps': self.ephys_sweeps,
            'neuronal_model': self.neuronal_model
            }

        return data



    
        
