# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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
