import os
import pandas as pd

from allensdk.config.model.manifest_builder import ManifestBuilder
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

        file_name = self.get_cache_path(file_name, 'CELLS')

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

        file_name = self.get_cache_path(file_name, 'EPHYS_FEATURES')

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

        file_name = self.get_cache_path(file_name, 'EPHYS_DATA', specimen_id)

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

        file_name = self.get_cache_path(file_name, 'RECONSTRUCTION', specimen_id)

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
        mb.add_path('CELLS', 'cells.json', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_DATA', 'specimen_%d/ephys.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path('EPHYS_FEATURES', 'ephys_features.csv', typename='file', parent_key='BASEDIR')
        mb.add_path('RECONSTRUCTION', 'specimen_%d/reconstruction.swc', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)

        




    
