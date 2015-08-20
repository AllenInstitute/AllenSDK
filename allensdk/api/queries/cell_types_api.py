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

from allensdk.api.queries.rma_api import RmaApi
import pandas as pd
import os

class CellTypesApi(RmaApi):

    def __init__(self, base_uri=None):
        super(CellTypesApi, self).__init__(base_uri)

        
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

        criteria = "[is_cell_specimen$eq'true'],products[name$eq'Mouse Cell Types']"
        
        include = ( 'structure,donor(transgenic_lines),specimen_tags,cell_soma_locations,' +
                    'ephys_features,data_sets,neuron_reconstructions' )

        cells = self.model_query('Specimen', criteria=criteria, include=include, num_rows='all')
        
        for cell in cells:
            # specimen tags
            for tag in cell['specimen_tags']:
                tag_name, tag_value = tag['name'].split(' - ')
                tag_name = tag_name.replace(' ','_')
                cell[tag_name] = tag_value


            # morphology and reconstuction
            cell['has_reconstruction'] = len(cell['neuron_reconstructions']) > 0
            cell['has_morphology'] = len(cell['data_sets']) > 0

        return self.filter_cells(cells, require_morphology, require_reconstruction)


    def get_ephys_sweeps(self, specimen_id):
        '''
        Query the API for a list of sweeps for a particular cell in the Cell Types Database.

        Parameters
        ----------

        specimen_id: int
            Specimen ID of a cell.

        Returns
        -------
        list: List of sweep dictionaries belonging to a cell
        '''
        criteria = "[specimen_id$eq%d]" % specimen_id
        
        return self.model_query('EphysSweep', criteria=criteria, num_rows='all')


    def filter_cells(self, cells, require_morphology, require_reconstruction):
        ''' 
        Filter a list of cell specimens to those that optionally have morphologies
        or have morphological reconstructions.

        Parameters
        ----------

        cells: list
            List of cell metadata dictionaries to be filtered

        require_morphology: boolean
            Filter out cells that have no morphological images.

        require_reconstruction: boolean
            Filter out cells that have no morphological reconstructions.
        '''

        if require_morphology:
            cells = [ c for c in cells if c['has_morphology'] ]

        if require_reconstruction:
            cells = [ c for c in cells if c['has_reconstruction'] ]

        return cells
        

    def get_ephys_features(self, dataframe=False):
        '''
        Query the API for the full table of EphysFeatures for all cells.  

        Parameters
        ----------
        
        dataframe: boolean
            If true, return the results as a Pandas DataFrame.  Otherwise
            return a list of dictionaries.
        '''

        features = self.model_query('EphysFeature', num_rows='all')

        if dataframe:
            return pd.DataFrame(features)
        else:
            return features


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

        try:
            file_url = results[0]['ephys_result']['well_known_files'][0]['download_link']
        except Exception, _:
            raise Exception("Specimen %d has no ephys data" % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
        

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



        

