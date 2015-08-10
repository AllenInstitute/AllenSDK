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

    def __init__(self, base_uri=None, cache=True, manifest_file='cell_types_manifest.json'):
        super(CellTypesApi, self).__init__(base_uri)

        self.cache = cache

        if self.cache:
            self.manifest = self.load_manifest(manifest_file)
        else:
            self.manifest = None


    def list_cells(self, require_morphology=False, require_reconstruction=False):
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
            file_name = self.manifest.get_path('CELLS')
        else:
            file_name = None

        if os.path.exists(file_name):
            cells = json_utilities.read(file_name)
        else:
            criteria = "[is_cell_specimen$eq'true'],products[name$eq'Mouse Cell Types']"

            include = ( 'structure,donor(transgenic_lines),specimen_tags,cell_soma_locations,' +
                        'ephys_features,data_sets,neuron_reconstructions' )

            cells = self.model_query('Specimen', criteria=criteria, include=include, num_rows='all')

            self.append_extra_cell_fields(cells)

            if self.cache:
                json_utilities.write(file_name, cells)

        if require_morphology:
            cells = [ c for c in cells if c['has_morphology'] ]

        if require_reconstruction:
            cells = [ c for c in cells if c['has_reconstruction'] ]

        return cells
            

    def append_extra_cell_fields(self, cells):
        for cell in cells:
            # specimen tags
            for tag in cell['specimen_tags']:
                tag_name, tag_value = tag['name'].split(' - ')
                tag_name = tag_name.replace(' ','_')
                cell[tag_name] = tag_value


            # morphology and reconstuction
            cell['has_reconstruction'] = len(cell['neuron_reconstructions']) > 0
            cell['has_morphology'] = len(cell['data_sets']) > 0
                

    def list_ephys_features(self, dataframe=False):
        if self.cache:
            file_name = self.manifest.get_path('EPHYS_FEATURES')
        else:
            file_name = None

        if os.path.exists(file_name):
            df = pd.DataFrame.from_csv(file_name)
            if dataframe:
                return df
            else:
                return df.to_dict('records')

        features = self.model_query('EphysFeature',
                                    num_rows='all')

        df = pd.DataFrame(features)
 
        if self.cache:
            df.to_csv(file_name)
            
        if dataframe:
            return df
        else:
            return df.to_dict('records')


    def save_ephys_data(self, specimen_id, file_name):
        try: 
            os.makedirs(os.path.dirname(file_name))
        except:
            pass

        criteria = '[id$eq%d],ephys_result(well_known_files(well_known_file_type[name$eqNWB]))' % specimen_id
        includes = 'ephys_result(well_known_files(well_known_file_type))'

        results = self.model_query('Specimen',
                                   criteria=criteria,
                                   include=includes,
                                   num_rows='all')

        file_url = results[0]['ephys_result']['well_known_files'][0]['download_link']

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
        

    def load_ephys_data(self, specimen_id, file_name=None):
        if not file_name and self.cache:
            file_name = self.manifest.get_path('EPHYS_DATA', specimen_id)
        else:
            raise Exception("Please enable caching (CellTypesApi.cache = True) or specify a save_file_name.")

        if not os.path.exists(file_name):
            self.save_ephys_data(specimen_id, file_name)

        return NwbDataSet(file_name)


    def save_reconstruction(self, specimen_id, file_name):
        try: 
            os.makedirs(os.path.dirname(file_name))
        except:
            pass

        criteria = '[id$eq%d],neuron_reconstructions(well_known_files)' % specimen_id
        includes = 'neuron_reconstructions(well_known_files)'
        
        results = self.model_query('Specimen',
                                   criteria=criteria,
                                   include=includes,
                                   num_rows='all')
        
        try:
            file_url = results[0]['neuron_reconstructions'][0]['well_known_files'][0]['download_link']
        except:
            raise Exception("Specimen %d has no reconstruction" % specimen_id)
        
        self.retrieve_file_over_http(self.api_url + file_url, file_name)


    def load_reconstruction(self, specimen_id, file_name=None):
        if not file_name and self.cache:
            file_name = self.manifest.get_path('RECONSTRUCTION', specimen_id)
        else:
            raise Exception("Please enable caching (CellTypesApi.cache = True) or specify a save_file_name.")

        if not os.path.exists(file_name):
            self.save_reconstruction(specimen_id, file_name)

        return swc.read_swc(file_name)


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

        
        

