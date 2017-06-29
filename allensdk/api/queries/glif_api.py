# Copyright 2015-2016 Allen Institute for Brain Science
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

import simplejson as json
import logging
from ...deprecated import deprecated
from .rma_template import RmaTemplate


class GlifApi(RmaTemplate):

    _log = logging.getLogger('allensdk.api.queries.glif_api')

    NWB_FILE_TYPE = None

    rma_templates = \
        {"glif_queries": [
            {'name': 'neuronal_model_templates',
             'description': 'see name',
             'model': 'NeuronalModelTemplate',
             'num_rows': 'all',
             'count': False,
             },
            {'name': 'neuronal_models',
             'description': 'see name',
             'model': 'Specimen',
             'include': 'neuronal_models(well_known_files,neuronal_model_template,neuronal_model_runs(well_known_files))',
             'criteria':'[id$in{{ ephys_experiment_ids }}]',
             'num_rows': 'all',
             'criteria_params':['ephys_experiment_ids'],
             'count': False,
             },
            {'name': 'neuron_config',
             'description': 'see name',
             'model': 'NeuronalModel',
             'include': 'well_known_files(well_known_file_type)',
             'criteria':'[id$in{{ neuronal_model_ids }}]',
             'num_rows': 'all',
             'criteria_params':['neuronal_model_ids'],
             'count': False,
            }
        ]
        }

    def __init__(self, base_uri=None):
        super(GlifApi, self).__init__(base_uri, query_manifest=GlifApi.rma_templates)

    def get_neuronal_model_templates(self):

        return self.template_query('glif_queries',
                                   'neuronal_model_templates')

    def get_neuronal_models(self, ephys_experiment_ids=None):
        return self.template_query('glif_queries',
                                   'neuronal_models', ephys_experiment_ids=ephys_experiment_ids)

    def get_neuronal_models_by_id(self, neuronal_model_ids=None):
        return self.template_query('glif_queries',
                                   'neuron_config', neuronal_model_ids=neuronal_model_ids)

    def get_neuron_configs(self, neuronal_model_ids=None):

        data = self.template_query('glif_queries',
                                   'neuron_config', neuronal_model_ids=neuronal_model_ids)

        return_dict = {}
        for curr_config in data:
            # print curr_config
            neuron_config_url = curr_config['well_known_files'][0]['download_link']
            return_dict[curr_config['id']] = self.retrieve_parsed_json_over_http(self.api_url +
                                                                                 neuron_config_url)

        return return_dict

    @deprecated()
    def list_neuronal_models(self):
        ''' DEPRECATED Query the API for a list of all GLIF neuronal models.

        Returns
        -------
        list
            Meta data for all GLIF neuronal models.
        '''

        include = "specimen(ephys_result[failed$eqfalse]),neuronal_model_template[name$il'*LIF*']"

        return self.model_query('NeuronalModel',
                                include=include,
                                num_rows='all')

    @deprecated()
    def get_neuronal_model(self, neuronal_model_id):
        '''DEPRECATED Query the current RMA endpoint with a neuronal_model id
        to get the corresponding well known files and meta data.

        Returns
        -------
        dict
            A dictionary containing
        '''


        include = ('neuronal_model_template(well_known_files(well_known_file_type)),' +
                   'specimen(ephys_sweeps,ephys_result(well_known_files(well_known_file_type))),' +
                   'well_known_files(well_known_file_type)')

        criteria = "[id$eq%d]" % neuronal_model_id

        self.neuronal_model = self.model_query('NeuronalModel',
                                               criteria=criteria,
                                               include=include,
                                               num_rows='all')[0]

        self.ephys_sweeps = None
        self.neuron_config_url = None
        self.stimulus_url = None

        # sweeps come from the specimen
        try:
            specimen = self.neuronal_model['specimen']
            self.ephys_sweeps = specimen['ephys_sweeps']
        except Exception as e:
            logging.info(e.args)
            self.ephys_sweeps = None

        if self.ephys_sweeps is None:
            logging.warning(
                "Could not find ephys_sweeps for this model (%d)" % self.neuronal_model['id'])

        # neuron config file comes from the neuronal model's well known files
        try:
            for wkf in self.neuronal_model['well_known_files']:
                if wkf['path'].endswith('neuron_config.json'):
                    self.neuron_config_url = wkf['download_link']
                    break
        except Exception as e:
            self.neuron_config_url = None

        if self.neuron_config_url is None:
            logging.warning(
                "Could not find neuron config well_known_file for this model (%d)" % self.neuronal_model['id'])

        # NWB file comes from the ephys_result's well known files
        try:
            ephys_result = specimen['ephys_result']
            for wkf in ephys_result['well_known_files']:
                if wkf['well_known_file_type']['name'] == 'NWBDownload':
                    self.stimulus_url = wkf['download_link']
                    break
        except Exception as e:
            self.stimulus_url = None

        if self.stimulus_url is None:
            logging.warning(
                "Could not find stimulus well_known_file for this model (%d)" % self.neuronal_model['id'])

        self.metadata = {
            'neuron_config_url': self.neuron_config_url,
            'stimulus_url': self.stimulus_url,
            'ephys_sweeps': self.ephys_sweeps,
            'neuronal_model': self.neuronal_model
        }

        return self.metadata

    @deprecated()
    def get_ephys_sweeps(self):
        ''' DEPRECATED Retrieve ephys sweep information out of downloaded metadata for a neuronal model

        Returns
        -------
        list
            A list of sweeps metadata dictionaries
        '''

        return self.ephys_sweeps

    @deprecated()
    def get_neuron_config(self, output_file_name=None):
        ''' DEPRECATED Retrieve a model configuration file from the API, optionally save it to disk, and
        return the contents of that file as a dictionary.

        Parameters
        ----------
        output_file_name: string
            File name to store the neuron configuration (optional).
        '''

        if self.neuron_config_url is None:
            raise Exception("URL for neuron config file is empty.")

        logging.info(self.api_url + self.neuron_config_url)

        neuron_config = self.retrieve_parsed_json_over_http(
            self.api_url + self.neuron_config_url)

        if output_file_name:
            with open(output_file_name, 'wb') as f:
                f.write(json.dumps(neuron_config, indent=2))

        return neuron_config

    @deprecated()
    def cache_stimulus_file(self, output_file_name):
        ''' DEPRECATED Download the NWB file for the current neuronal model and save it to a file.

        Parameters
        ----------
        output_file_name: string
            File name to store the NWB file.
        '''

        if self.stimulus_url is None:
            raise Exception("URL for stimulus file is empty.")

        self.retrieve_file_over_http(
            self.api_url + self.metadata['stimulus_url'], output_file_name)
