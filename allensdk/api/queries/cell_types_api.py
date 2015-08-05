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

import os, logging

import csv
import pandas as pd

from allensdk.api.queries.rma.rma_simple_api import RmaSimpleApi
from allensdk.config.model.manifest_builder import ManifestBuilder
from allensdk.config.model.manifest import Manifest

import allensdk.core.json_utilities as json_utilities
from allensdk.core.nwb_data_set import NwbDataSet

import allensdk.core.swc as swc


class CellTypesApi(RmaSimpleApi):
    def __init__(self, base_uri=None, cache=True, manifest_file='manifest.json'):
        super(CellTypesApi, self).__init__(base_uri)

        self.cache = cache

        if self.cache:
            self.manifest = self.load_manifest(manifest_file)
        else:
            self.manifest = None


    def list_cells(self, require_morphology=False, require_reconstruction=False, recache=False):
        ''' Query the API for a list of all cells in the Cell Types Database.

        Parameters
        ----------
        require_morphology: boolean
            Only return cells that have morphology images.

        require_reconstruction: boolean
            Only return cells that have morphological reconstructions.

        Returns
        -------
        list
            Meta data for all cells.
        '''

        if self.cache:
            path = self.manifest.get_path('CELLS')
        else:
            path = None

        if not recache and os.path.exists(path):
            return json_utilities.read(path)

        criteria = "[is_cell_specimen$eq'true'],products[name$eq'Mouse Cell Types']"

        if require_morphology:
            criteria += ',data_sets'
            
        if require_reconstruction:
            criteria += ',neuron_reconstructions'

        cells = self.model_query('Specimen',
                                 criteria=criteria,
                                 include='structure,donor(transgenic_lines),specimen_tags,cell_soma_locations,ephys_features',
                                 num_rows='all')

        self.parse_tags(cells)

        if self.cache:
            json_utilities.write(path, cells)
        
        return cells
            

    def parse_tags(self, cells):
        for cell in cells:
            for tag in cell['specimen_tags']:
                tag_name, tag_value = tag['name'].split(' - ')
                tag_name = tag_name.replace(' ','_')
                cell[tag_name] = tag_value
                

    def list_ephys_features(self, dataframe=False):
        if self.cache:
            path = self.manifest.get_path('EPHYS_FEATURES')
        else:
            path = None

        if os.path.exists(path):
            df = pd.DataFrame.from_csv(path)
            if dataframe:
                return df
            else:
                return df.to_dict('records')

        features = self.model_query('EphysFeature',
                                    num_rows='all')

        df = pd.DataFrame(features)
 
        if self.cache:
            df.to_csv(path)
            
        if dataframe:
            return df
        else:
            return df.to_dict('records')


    def load_ephys_data(self, specimen_id, save_file_name=None):
        if save_file_name:
            path = save_file_name
        elif self.cache:
            path = self.manifest.get_path('EPHYS_DATA', specimen_id)
        else:
            raise Exception("Please enable caching (CellTypesApi.cache = True) or specify a save_file_name.")

        if not os.path.exists(path):
            try: 
                os.makedirs(os.path.dirname(path))
            except:
                pass

            criteria = '[id$eq%d],ephys_result(well_known_files(well_known_file_type[name$eqNWB]))' % specimen_id
            includes = 'ephys_result(well_known_files(well_known_file_type))'

            results = self.model_query('Specimen',
                                       criteria=criteria,
                                       include=includes,
                                       num_rows='all')
            file_url = results[0]['ephys_result']['well_known_files'][0]['download_link']

            self.retrieve_file_over_http(self.api_url + file_url, path)

        return NwbDataSet(path)


    def load_reconstruction(self, specimen_id, save_file_name=None):
        if save_file_name:
            path = save_file_name
        elif self.cache:
            path = self.manifest.get_path('RECONSTRUCTION', specimen_id)
        else:
            raise Exception("Please enable caching (CellTypesApi.cache = True) or specify a save_file_name.")

        if not os.path.exists(path):
            try: 
                os.makedirs(os.path.dirname(path))
            except:
                pass

            criteria = '[id$eq%d],neuron_reconstructions(well_known_files)' % specimen_id
            includes = 'neuron_reconstructions(well_known_files)'

            results = self.model_query('Specimen',
                                       criteria=criteria,
                                       include=includes,
                                       num_rows='all')

            file_url = results[0]['neuron_reconstructions'][0]['well_known_files'][0]['download_link']

            self.retrieve_file_over_http(self.api_url + file_url, path)

        return swc.read_swc(path)


    def load_manifest(self, file_name):
        if not os.path.exists(file_name):
            self.build_manifest(file_name)

        manifest_json = json_utilities.read(file_name)
        manifest = Manifest(manifest_json['manifest'], 
                            os.path.dirname(file_name))

        return manifest
            

    def build_manifest(self, file_name):
        mb = ManifestBuilder()

        mb.add_path('BASEDIR', '.')
        mb.add_path('CELLS', 'cells.json', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_DATA', 'specimen_%d/ephys.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_FEATURES', 'ephys_features.csv', typename='file', parent_key='BASEDIR')
        mb.add_path('RECONSTRUCTION', 'specimen_%d/reconstruction.swc', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)

        
        

