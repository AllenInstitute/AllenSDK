import os
import pandas as pd

from allensdk.config.model.manifest_builder import ManifestBuilder
from allensdk.api.cache import Cache
from allensdk.api.queries.cell_types_api import CellTypesApi

import allensdk.core.json_utilities as json_utilities
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.core.swc as swc

class CellTypes(Cache):
    def __init__(self, cache=True, manifest_file='manifest.json'):

        super(CellTypes, self).__init__(manifest=manifest_file, cache=cache)

        self.api = CellTypesApi()


    def get_cells(self, require_morphology=False, require_reconstruction=False):
        file_name = self.get_cache_path(None, 'CELLS')

        if os.path.exists(file_name):
            cells = json_utilities.read(file_name)
        else:
            cells = self.api.get_cells(False, False)

            if self.cache:
                json_utilities.write(file_name, cells)

        # filter the cells on the way out
        return self.api.filter_cells(cells, require_morphology, require_reconstruction)


    def get_ephys_features(self, dataframe=False):
        file_name = self.get_cache_path(None, 'EPHYS_FEATURES')

        if os.path.exists(file_name):
            features_df = pd.DataFrame.from_csv(file_name)
        else:
            features_df = self.api.get_ephys_features(dataframe=True)

            if self.cache:
                features_df.to_csv(file_name)

        if dataframe:
            return features_df
        else:
            return features_df.to_dict('records')


    def get_ephys_data(self, specimen_id, file_name=None):
        file_name = self.get_cache_path(file_name, 'EPHYS_DATA', specimen_id)

        if not os.path.exists(file_name):
            self.api.save_ephys_data(specimen_id, file_name)

        return NwbDataSet(file_name)


    def get_reconstruction(self, specimen_id, file_name=None):
        file_name = self.get_cache_path(file_name, 'RECONSTRUCTION', specimen_id)

        if file_name is None:
            raise Exception("Please enable caching (CellTypes.cache = True) or specify a save_file_name.")

        if not os.path.exists(file_name):
            self.api.save_reconstruction(specimen_id, file_name)

        return swc.read_swc(file_name)


    def build_manifest(self, file_name):
        mb = ManifestBuilder()

        mb.add_path('BASEDIR', '.')
        mb.add_path('CELLS', 'cells.json', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_DATA', 'specimen_%d/ephys.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_FEATURES', 'ephys_features.csv', typename='file', parent_key='BASEDIR')
        mb.add_path('RECONSTRUCTION', 'specimen_%d/reconstruction.swc', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)

        




    
