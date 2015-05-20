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

import os, json, logging, urllib

class CellTypesApi(Api):
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
        
        return self.do_rma_query(self.build_list_cells_rma, self.read_list_cells_json, 
                                 require_morphology, require_reconstruction)


    def build_list_cells_rma(self, require_morphology, require_reconstruction, fmt='json'):
        ''' Build the RMA URL that will fetch meta data for all cells in the cell types database '''

        criteria_associations = "[is_cell_specimen$eq'true'],products[name$eq'Mouse Cell Types']"

        if require_morphology:
            criteria_associations += ",data_sets"

        if require_reconstruction:
            criteria_associations += ",neuron_reconstructions"

        include_associations = 'structure,donor(transgenic_lines),specimen_tags,cell_soma_locations'

        url = ''.join([self.rma_endpoint, 
                       '/query.',
                       fmt,
                       '?q=',
                       'model::Specimen,',
                       'rma::criteria,',
                       criteria_associations,
                       ',rma::include,',
                       include_associations,
                       ',rma::options[num_rows$eqall]'])

        return url


    def read_list_cells_json(self, parsed_json):
        ''' Return the list of cells from the parsed query. '''
        return parsed_json['msg']


    def save_ephys_data(self, specimen_id, file_name):
        ''' Find the electrophysiology data file for a specimen and save it. '''

        
        file_url = self.do_rma_query(self.build_ephys_data_rma, self.read_ephys_data_json, specimen_id)

        if file_url is None:
            raise Exception("Could not find electrophysiology file for specimen %d " % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
    

    def build_ephys_data_rma(self, specimen_id, fmt='json'):
        ''' Build the URL for the save_ephys_data command. '''
        criteria = '[id$eq%d],ephys_result(well_known_files(well_known_file_type[name$eqNWB]))' % specimen_id
        includes = 'ephys_result(well_known_files(well_known_file_type))'

        url = ''.join([self.rma_endpoint, 
                       '/query.',
                       fmt,
                       '?q=',
                       'model::Specimen,',
                       'rma::criteria,',
                       criteria,
                       ',rma::include,',
                       includes,
                       ',rma::options[num_rows$eqall]'])

        return url


    def read_ephys_data_json(self, parsed_json):
        ''' Extract the download link from the save_ephys_data RMA. '''

        try:
            return parsed_json['msg'][0]['ephys_result']['well_known_files'][0]['download_link']
        except Exception, e:
            logging.error("Error finding ephys file: %s" % e.message)
            return None


    def save_reconstruction(self, specimen_id, file_name):
        ''' Find the reconstruction file for a specimen and save it. '''

        file_url = self.do_rma_query(self.build_reconstruction_rma, self.read_reconstruction_json, specimen_id)

        if file_url is None:
            raise Exception("Could not find reconstruction file for specimen %d " % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)


    def build_reconstruction_rma(self, specimen_id, fmt='json'):
        ''' Build the URL for the save_reconstruction command. '''
        criteria = '[id$eq%d],neuron_reconstructions(well_known_files)' % specimen_id
        includes = 'neuron_reconstructions(well_known_files)'

        url = ''.join([self.rma_endpoint, 
                       '/query.',
                       fmt,
                       '?q=',
                       'model::Specimen,',
                       'rma::criteria,',
                       criteria,
                       ',rma::include,',
                       includes,
                       ',rma::options[num_rows$eqall]'])

        return url


    def read_reconstruction_json(self, parsed_json):
        ''' Extract the download link from the save_ephys_file RMA. '''

        try:
            return parsed_json['msg'][0]['neuron_reconstructions'][0]['well_known_files'][0]['download_link']
        except Exception, e:
            logging.error("Error finding reconstruction file: %s" % e.message)
            return None
    
