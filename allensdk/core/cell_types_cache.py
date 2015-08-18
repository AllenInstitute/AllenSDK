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

import os
import pandas as pd

from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.api.cache import Cache
from allensdk.api.queries.cell_types_api import CellTypesApi

import allensdk.core.json_utilities as json_utilities
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.core.swc as swc

class CellTypesCache(Cache):
    """
    Cache class for storing and accessing data from the Cell Types Database.
    By default, this class will cache any downloaded metadata or files in 
    well known locations defined in a manifest file.  This behavior can be
    disabled.

    Attributes
    ----------
    
    api: CellTypesApi instance
        The object used for making API queries related to the Cell Types Database

    Parameters
    ----------
    
    cache: boolean
        Whether the class should save results of API queries to locations specified
        in the manifest file.  Queries for files (as opposed to metadata) must have a
        file location.  If caching is disabled, those locations must be specified
        in the function call (e.g. get_ephys_data(file_name='file.nwb')).

    manifest_file: string
       File name of the manifest to be read.  Default is "manifest.json".
    """

    CELLS_KEY = 'CELLS'
    EPHYS_FEATURES_KEY = 'EPHYS_FEATURES'
    EPHYS_DATA_KEY = 'EPHYS_DATA'
    RECONSTRUCTION_KEY = 'RECONSTRUCTION'
    
    def __init__(self, cache=True, manifest_file='manifest.json'):
        super(CellTypesCache, self).__init__(manifest=manifest_file, cache=cache)
        self.api = CellTypesApi()


    def get_cells(self, file_name=None, require_morphology=False, require_reconstruction=False):
        """
        Download metadata for all cells in the database and optionally return a
        subset filtered by whether or not they have a morphology or reconstruction.

        Parameters
        ----------
        
        file_name: string
            File name to save/read the cell metadata as JSON.  If file_name is None, 
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        require_morphology: boolean
            Filter out cells that have no morphological images.

        require_reconstruction: boolean
            Filter out cells that hve no morphological reconstructions.
        """

        file_name = self.get_cache_path(file_name, self.CELLS_KEY)

        if os.path.exists(file_name):
            cells = json_utilities.read(file_name)
        else:
            cells = self.api.list_cells(False, False)

            if self.cache:
                json_utilities.write(file_name, cells)

        # filter the cells on the way out
        return self.api.filter_cells(cells, require_morphology, require_reconstruction)


    def get_ephys_features(self, dataframe=False, file_name=None):
        """
        Download electrophysiology features for all cells in the database.

        Parameters
        ----------
        
        file_name: string
            File name to save/read the ephys features metadata as CSV.  
            If file_name is None, the file_name will be pulled out of the 
            manifest.  If caching is disabled, no file will be saved. 
            Default is None.

        dataframe: boolean
            Return the output as a Pandas DataFrame.  If False, return 
            a list of dictionaries.
        """

        file_name = self.get_cache_path(file_name, self.EPHYS_FEATURES_KEY)

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
        """
        Download electrophysiology traces for a single cell in the database.

        Parameters
        ----------
        
        specimen_id: int
            The ID of a cell specimen to download.

        file_name: string
            File name to save/read the ephys features metadata as CSV.  
            If file_name is None, the file_name will be pulled out of the 
            manifest.  If caching is disabled, no file will be saved. 
            Default is None.

        Returns
        -------
        NwbDataSet
            A class instance with helper methods for retrieving stimulus
            and response traces out of an NWB file.
        """

        file_name = self.get_cache_path(file_name, self.EPHYS_DATA_KEY, specimen_id)

        if not os.path.exists(file_name):
            self.api.save_ephys_data(specimen_id, file_name)

        return NwbDataSet(file_name)


    def get_reconstruction(self, specimen_id, file_name=None):
        """
        Download and open a reconstruction for a single cell in the database.

        Parameters
        ----------
        
        specimen_id: int
            The ID of a cell specimen to download.

        file_name: string
            File name to save/read the ephys features metadata as CSV.  
            If file_name is None, the file_name will be pulled out of the 
            manifest.  If caching is disabled, no file will be saved. 
            Default is None.

        Returns
        -------
        Morphology
             A class instance with methods for accessing morphology compartments.
        """

        file_name = self.get_cache_path(file_name, self.RECONSTRUCTION_KEY, specimen_id)

        if file_name is None:
            raise Exception("Please enable caching (CellTypes.cache = True) or specify a save_file_name.")

        if not os.path.exists(file_name):
            self.api.save_reconstruction(specimen_id, file_name)

        return swc.read_swc(file_name)


    def build_manifest(self, file_name):
        """
        Construct a manifest for this Cache class and save it in a file.
        
        Parameters
        ----------
        
        file_name: string
            File location to save the manifest.

        """

        mb = ManifestBuilder()

        mb.add_path('BASEDIR', '.')
        mb.add_path(self.CELLS_KEY, 'cells.json', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EPHYS_DATA_KEY, 'specimen_%d/ephys.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EPHYS_FEATURES_KEY, 'ephys_features.csv', typename='file', parent_key='BASEDIR')
        mb.add_path(self.RECONSTRUCTION_KEY, 'specimen_%d/reconstruction.swc', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)

        




    
