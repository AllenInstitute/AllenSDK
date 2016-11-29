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

from .rma_api import RmaApi


class GridDataApi(RmaApi):
    '''HTTP Client for the Allen 3-D Expression Grid Data Service.

    See: `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data>`_
    '''

    INJECTION_DENSITY = 'injection_density'
    PROJECTION_DENSITY = 'projection_density'
    INJECTION_FRACTION = 'injection_fraction'
    INJECTION_ENERGY = 'injection_energy'
    PROJECTION_ENERGY = 'projection_energy'
    DATA_MASK = 'data_mask'

    ENERGY = 'energy'
    DENSITY = 'density'
    INTENSITY = 'intensity'

    def __init__(self,
                 resolution=None,
                 base_uri=None):
        super(GridDataApi, self).__init__(base_uri)

        if resolution is None:
            resolution = 25
        self.resolution = resolution

    def download_expression_grid_data(self,
                                      section_data_set_id,
                                      include=None,
                                      path=None):
        '''Download in NRRD format.

        Parameters
        ----------
        section_data_set_id : integer
            What to download.
        include : list of strings, optional
            Image volumes. 'energy' (default), 'density', 'intensity'.
        path : string, optional
            File name to save as.
        i
        Returns
        -------
            file : 3-D expression grid data packaged into a compressed archive file (.zip).

        Notes
        -----
        '''
        if include is not None:
            include_clause = ''.join(['?include=',
                                      ','.join(include)])
        else:
            include_clause = ''

        url = ''.join([self.grid_data_endpoint,
                       '/download/',
                       str(section_data_set_id),
                       include_clause])

        if path is None:
            path = str(section_data_set_id) + '.zip'

        self.retrieve_file_over_http(url, path)

    def download_projection_grid_data(self,
                                      section_data_set_id,
                                      image=None,
                                      resolution=None,
                                      save_file_path=None):
        '''Download in NRRD format.

        Parameters
        ----------
        section_data_set_id : integer
            What to download.
        image : list of strings, optional
            Image volume. 'projection_density', 'projection_energy', 'injection_fraction', 'injection_density', 'injection_energy', 'data_mask'.
        resolution : integer, optional
            in microns. 10, 25, 50, or 100 (default).
        save_file_path : string, optional
            File name to save as.

        Notes
        -----
        See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#name="Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA">`_
        for additional documentation.
        '''
        params_list = []

        if image is not None:
            params_list.append('image=' + ','.join(image))

        if resolution is not None:
            params_list.append('resolution=%d' % (resolution))

        if len(params_list) > 0:
            params_clause = '?' + '&'.join(params_list)
        else:
            params_clause = ''

        url = ''.join([self.grid_data_endpoint,
                       '/download_file/',
                       str(section_data_set_id),
                       params_clause])

        if save_file_path is None:
            save_file_path = str(section_data_set_id) + '.nrrd'

        self.retrieve_file_over_http(url, save_file_path)
