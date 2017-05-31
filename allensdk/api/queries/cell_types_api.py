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

from .rma_api import RmaApi
from ..cache import cacheable
from allensdk.config.manifest import Manifest
from allensdk.api.cache import Cache


class CellTypesApi(RmaApi):
    NWB_FILE_TYPE = 'NWBDownload'
    SWC_FILE_TYPE = '3DNeuronReconstruction'
    MARKER_FILE_TYPE = '3DNeuronMarker'

    def __init__(self, base_uri=None):
        super(CellTypesApi, self).__init__(base_uri)

    def list_cells(self,
                   require_morphology=False,
                   require_reconstruction=False,
                   reporter_status=None):
        """
        Query the API for a list of all cells in the Cell Types Database.

        Parameters
        ----------
        require_morphology: boolean
            Only return cells that have morphology images.

        require_reconstruction: boolean
            Only return cells that have morphological reconstructions.

        reporter_status: list
            Return cells that have a particular cell reporter status.

        Returns
        -------
        list
            Meta data for all cells.
        """

        criteria = "[is_cell_specimen$eq'true'],products[name$eq'Mouse Cell Types'],ephys_result[failed$eqfalse]"

        include = ('structure,donor(transgenic_lines),specimen_tags,cell_soma_locations,' +
                   'ephys_features,data_sets,neuron_reconstructions,cell_reporter')

        cells = self.model_query(
            'Specimen', criteria=criteria, include=include, num_rows='all')

        for cell in cells:
            # specimen tags
            for tag in cell['specimen_tags']:
                tag_name, tag_value = tag['name'].split(' - ')
                tag_name = tag_name.replace(' ', '_')
                cell[tag_name] = tag_value

            # morphology and reconstuction
            cell['has_reconstruction'] = len(
                cell['neuron_reconstructions']) > 0
            cell['has_morphology'] = len(cell['data_sets']) > 0

            # transgenic line
            cell['transgenic_line'] = None
            for tl in cell['donor']['transgenic_lines']:
                if tl['transgenic_line_type_name'] == 'driver':
                    cell['transgenic_line'] = tl['name']

            # cell reporter status
            cell['reporter_status'] = cell['cell_reporter']['name']

        return self.filter_cells(cells, require_morphology, require_reconstruction, reporter_status)

    @cacheable()
    def get_ephys_sweeps(self, specimen_id):
        """
        Query the API for a list of sweeps for a particular cell in the Cell Types Database.

        Parameters
        ----------

        specimen_id: int
            Specimen ID of a cell.

        Returns
        -------
        list: List of sweep dictionaries belonging to a cell
        """
        criteria = "[specimen_id$eq%d]" % specimen_id
        sweeps = self.model_query(
            'EphysSweep', criteria=criteria, num_rows='all')
        return sorted(sweeps, key=lambda x: x['sweep_number'])

    def filter_cells(self, cells, require_morphology, require_reconstruction, reporter_status):
        """
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

        reporter_status: list
            Filter for cells that have a particular cell reporter status
        """

        if require_morphology:
            cells = [c for c in cells if c['has_morphology']]

        if require_reconstruction:
            cells = [c for c in cells if c['has_reconstruction']]

        if reporter_status:
            cells = [c for c in cells if c[
                'reporter_status'] in reporter_status]

        return cells

    @cacheable()
    def get_ephys_features(self):
        """
        Query the API for the full table of EphysFeatures for all cells.
        """

        return self.model_query(
            'EphysFeature',
            criteria='specimen(ephys_result[failed$eqfalse])',
            num_rows='all')

    @cacheable()
    def get_morphology_features(self):
        """
        Query the API for the full table of morphology features for all cells
        
        Notes
        -----
        by default the tags column is removed because it isn't useful
        """
        return self.model_query(
            'NeuronReconstruction',
            criteria="specimen(ephys_result[failed$eqfalse])",
            excpt='tags',
            num_rows='all')

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=2,
                                           path_keyword='file_name'))
    def save_ephys_data(self, specimen_id, file_name):
        """
        Save the electrophysology recordings for a cell as an NWB file.

        Parameters
        ----------
        specimen_id: int
            ID of the specimen, from the Specimens database model in the Allen Institute API.

        file_name: str
            Path to save the NWB file.
        """
        criteria = '[id$eq%d],ephys_result(well_known_files(well_known_file_type[name$eq%s]))' % (
            specimen_id, self.NWB_FILE_TYPE)
        includes = 'ephys_result(well_known_files(well_known_file_type))'

        results = self.model_query('Specimen',
                                   criteria=criteria,
                                   include=includes,
                                   num_rows='all')

        try:
            file_url = results[0]['ephys_result'][
                'well_known_files'][0]['download_link']
        except Exception as _:
            raise Exception("Specimen %d has no ephys data" % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)

    def save_reconstruction(self, specimen_id, file_name):
        """
        Save the morphological reconstruction of a cell as an SWC file.

        Parameters
        ----------
        specimen_id: int
            ID of the specimen, from the Specimens database model in the Allen Institute API.

        file_name: str
            Path to save the SWC file.
        """

        Manifest.safe_make_parent_dirs(file_name)

        criteria = '[id$eq%d],neuron_reconstructions(well_known_files)' % specimen_id
        includes = 'neuron_reconstructions(well_known_files(well_known_file_type[name$eq\'%s\']))' % self.SWC_FILE_TYPE

        results = self.model_query('Specimen',
                                   criteria=criteria,
                                   include=includes,
                                   num_rows='all')

        try:
            file_url = results[0]['neuron_reconstructions'][
                0]['well_known_files'][0]['download_link']
        except:
            raise Exception("Specimen %d has no reconstruction" % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)

    def save_reconstruction_markers(self, specimen_id, file_name):
        """
        Save the marker file for the morphological reconstruction of a cell.  These are
        comma-delimited files indicating points of interest in a reconstruction (truncation
        points, early tracing termination, etc).

        Parameters
        ----------
        specimen_id: int
            ID of the specimen, from the Specimens database model in the Allen Institute API.

        file_name: str
            Path to save the marker file.
        """

        Manifest.safe_make_parent_dirs(file_name)

        criteria = '[id$eq%d],neuron_reconstructions(well_known_files)' % specimen_id
        includes = 'neuron_reconstructions(well_known_files(well_known_file_type[name$eq\'%s\']))' % self.MARKER_FILE_TYPE

        results = self.model_query('Specimen',
                                   criteria=criteria,
                                   include=includes,
                                   num_rows='all')

        try:
            file_url = results[0]['neuron_reconstructions'][
                0]['well_known_files'][0]['download_link']
        except:
            raise LookupError("Specimen %d has no marker file" % specimen_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
