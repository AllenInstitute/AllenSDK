# Copyright 2016 Allen Institute for Brain Science
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

from ..api import Api
import os
import json
from collections import OrderedDict
from allensdk.config.manifest import Manifest


class BiophysicalApi(Api):
    _NWB_file_type = 'NWBDownload'
    _SWC_file_type = '3DNeuronReconstruction'
    _MOD_file_type = 'BiophysicalModelDescription'
    _FIT_file_type = 'NeuronalModelParameters'
    _MARKER_file_type = '3DNeuronMarker'

    def __init__(self, base_uri=None):
        super(BiophysicalApi, self).__init__(base_uri)
        self.cache_stimulus = True
        self.ids = {}
        self.sweeps = []
        self.manifest = {}
        self.model_type = None

    def build_rma(self, neuronal_model_id, fmt='json'):
        '''Construct a query to find all files related to a neuronal model.

        Parameters
        ----------
        neuronal_model_id : integer or string representation
            key of experiment to retrieve.
        fmt : string, optional
            json (default) or xml

        Returns
        -------
        string
            RMA query url.
        '''
        include_associations = ''.join([
            'neuronal_model_template(well_known_files(well_known_file_type)),',
            'specimen(ephys_result(well_known_files(well_known_file_type)),',
            'neuron_reconstructions(well_known_files(well_known_file_type)),',
            'ephys_sweeps),',
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
        '''Get the list of well_known_file ids from a response body
        containing nested sample,microarray_slides,well_known_files.

        Parameters
        ----------
        json_parsed_data : dict
           Response from the Allen Institute Api RMA.

        Returns
        -------
        list of strings
            Well known file ids.
        '''
        self.ids = {
            'stimulus': {},
            'morphology': {},
            'marker': {},
            'modfiles': {},
            'fit': {}
        }
        self.sweeps = []

        if 'msg' in json_parsed_data:
            for neuronal_model in json_parsed_data['msg']:
                if 'well_known_files' in neuronal_model:
                    for well_known_file in neuronal_model['well_known_files']:
                        if ('id' in well_known_file and
                            'path' in well_known_file and
                            self.is_well_known_file_type(well_known_file,
                                                         BiophysicalApi._FIT_file_type)):
                            self.ids['fit'][str(well_known_file['id'])] = \
                                os.path.split(well_known_file['path'])[1]

                if 'neuronal_model_template' in neuronal_model:
                    neuronal_model_template = neuronal_model[
                        'neuronal_model_template']
                    self.model_type = neuronal_model_template['name']
                    if 'well_known_files' in neuronal_model_template:
                        for well_known_file in neuronal_model_template['well_known_files']:
                            if ('id' in well_known_file and
                                'path' in well_known_file and
                                self.is_well_known_file_type(well_known_file,
                                                             BiophysicalApi._MOD_file_type)):
                                self.ids['modfiles'][str(well_known_file['id'])] = \
                                    os.path.join('modfiles',
                                                 os.path.split(well_known_file['path'])[1])

                if 'specimen' in neuronal_model:
                    specimen = neuronal_model['specimen']

                    if 'neuron_reconstructions' in specimen:
                        for neuron_reconstruction in specimen['neuron_reconstructions']:
                            if 'well_known_files' in neuron_reconstruction:
                                for well_known_file in neuron_reconstruction['well_known_files']:
                                    if ('id' in well_known_file and 'path' in well_known_file):
                                        if self.is_well_known_file_type(well_known_file, BiophysicalApi._SWC_file_type):
                                            self.ids['morphology'][str(well_known_file['id'])] = \
                                                os.path.split(
                                                    well_known_file['path'])[1]
                                        elif self.is_well_known_file_type(well_known_file, BiophysicalApi._MARKER_file_type):
                                            self.ids['marker'][str(well_known_file['id'])] = \
                                                os.path.split(
                                                    well_known_file['path'])[1]

                    if 'ephys_result' in specimen:
                        ephys_result = specimen['ephys_result']
                        if 'well_known_files' in ephys_result:
                            for well_known_file in ephys_result['well_known_files']:
                                if ('id' in well_known_file and
                                    'path' in well_known_file and
                                    self.is_well_known_file_type(well_known_file, BiophysicalApi._NWB_file_type)):
                                        self.ids['stimulus'][str(well_known_file['id'])] = \
                                            "%d.nwb" % (ephys_result['id'])

                    self.sweeps = [sweep['sweep_number']
                                   for sweep in specimen['ephys_sweeps']
                                   if sweep['stimulus_name'] != 'Test']

        return self.ids

    def is_well_known_file_type(self, wkf, name):
        '''Check if a structure has the expected name.

        Parameters
        ----------
        wkf : dict
            A well-known-file structure with nested type information.
        name : string
            The expected type name

        See Also
        --------
        read_json: where this helper function is used.
        '''
        try:
            return wkf['well_known_file_type']['name'] == name
        except:
            return False

    def get_well_known_file_ids(self, neuronal_model_id):
        '''Query the current RMA endpoint with a neuronal_model id
        to get the corresponding well known file ids.

        Returns
        -------
        list
            A list of well known file id strings.
        '''
        rma_builder_fn = self.build_rma
        json_traversal_fn = self.read_json

        return self.do_query(rma_builder_fn, json_traversal_fn, neuronal_model_id)

    def create_manifest(self,
                        fit_path='',
                        model_type='',
                        stimulus_filename='',
                        swc_morphology_path='',
                        marker_path='',
                        sweeps=[]):
        '''Generate a json configuration file with parameters for a
        a biophysical experiment.

        Parameters
        ----------
        fit_path : string
            filename of a json configuration file with cell parameters.
        stimulus_filename : string
            path to an NWB file with input currents.
        swc_morphology_path : string
            file in SWC format.
        sweeps : array of integers
            which sweeps in the stimulus file are to be used.
        '''
        self.manifest = OrderedDict()
        self.manifest['biophys'] = [{
            'model_file': ['manifest.json', fit_path],
            'model_type': model_type
        }]
        self.manifest['runs'] = [{
            'sweeps': sweeps
        }]
        self.manifest['neuron'] = [{
            'hoc': ['stdgui.hoc', 'import3d.hoc']
        }]
        self.manifest['manifest'] = [
            {
                'type': 'dir',
                'spec': '.',
                'key': 'BASEDIR'
            },
            {
                'type': 'dir',
                'spec': 'work',
                'key': 'WORKDIR',
                'parent': 'BASEDIR'
            },
            {
                'type': 'file',
                'spec': swc_morphology_path,
                'key': 'MORPHOLOGY'
            },
            {
                'type': 'file',
                'spec': marker_path,
                'key': 'MARKER'
            },
            {
                'type': 'dir',
                'spec': 'modfiles',
                'key': 'MODFILE_DIR'
            },
            {
                'type': 'file',
                'format': 'NWB',
                'spec': stimulus_filename,
                'key': 'stimulus_path'
            },
            {
                'parent_key': 'WORKDIR',
                'type': 'file',
                'format': 'NWB',
                'spec': stimulus_filename,
                'key': 'output_path'
            }
        ]

    def cache_data(self,
                   neuronal_model_id,
                   working_directory=None):
        '''Take a an experiment id, query the Api RMA to get well-known-files
        download the files, and store them in the working directory.

        Parameters
        ----------
        neuronal_model_id : int or string representation
            found in the neuronal_model table in the api
        working_directory : string
            Absolute path name where the downloaded well-known files will be stored.
        '''
        if working_directory is None:
            working_directory = self.default_working_directory

        well_known_file_id_dict = self.get_well_known_file_ids(
            neuronal_model_id)

        if not well_known_file_id_dict or \
           (not any(well_known_file_id_dict.values())):
            raise(Exception("No data found for neuronal model id %d" %
                            (neuronal_model_id)))

        Manifest.safe_mkdir(working_directory)

        work_dir = os.path.join(working_directory, 'work')
        Manifest.safe_mkdir(work_dir)

        modfile_dir = os.path.join(working_directory, 'modfiles')
        Manifest.safe_mkdir(modfile_dir)

        for key, id_dict in well_known_file_id_dict.items():
            if (not self.cache_stimulus) and (key == 'stimulus'):
                continue

            for well_known_id, filename in id_dict.items():
                well_known_file_url = self.construct_well_known_file_download_url(
                    well_known_id)
                cached_file_path = os.path.join(working_directory, filename)
                self.retrieve_file_over_http(
                    well_known_file_url, cached_file_path)

        fit_path = self.ids['fit'].values()[0]
        stimulus_filename = self.ids['stimulus'].values()[0]
        swc_morphology_path = self.ids['morphology'].values()[0]
        marker_path = \
            self.ids['marker'].values()[0] if 'marker' in self.ids else ''
        sweeps = sorted(self.sweeps)

        self.create_manifest(fit_path,
                             self.model_type,
                             stimulus_filename,
                             swc_morphology_path,
                             marker_path,
                             sweeps)

        manifest_path = os.path.join(working_directory, 'manifest.json')
        with open(manifest_path, 'wb') as f:
            f.write(json.dumps(self.manifest, indent=2))
