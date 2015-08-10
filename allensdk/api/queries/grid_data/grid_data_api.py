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

from allensdk.api.queries.rma.rma_api import RmaApi
from allensdk.api.cache import Cache
import numpy as np
import nrrd


class GridDataApi(RmaApi, Cache):
    '''HTTP Client for the Allen 3-D Expression Grid Data Service.
    
    See: `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data>`_
    '''
    DEFAULT_RESOLUTION = 25

    INJECTION_DENSITY = 'injection_density'
    PROJECTION_DENSITY = 'projection_density'
    INJECTION_FRACTION = 'injection_fraction'
    INJECTION_ENERGY = 'injection_energy'
    PROJECTION_ENERGY = 'projection_energy'
    DATA_MASK = 'DATA_MASK'
    
    ENERGY='energy'
    DENSITY='density'
    INTENSITY='intensity'
    
    def __init__(self,
                 resolution=None,
                 base_uri=None,
                 cache=False):
        super(GridDataApi, self).__init__(base_uri)
        Cache.__init__(self, cache=cache)
        
        if resolution == None:
            resolution = GridDataApi.DEFAULT_RESOLUTION
            
        self.resolution = resolution

        
    def cache_expression_grid_data(self,
                                   experiment_id,
                                   include=None,
                                   path=None
                                   ):
        if type(include) is not list:
            include = [include]
            
        if self.cache == True:        
            self.download_expression_grid_data(
                experiment_id, include=include, path=path)
        
    
    def cache_projection_grid_data(self,
                                   path,
                                   eid,
                                   image=None,
                                   resolution=None):
        if type(image) is not list:
            image = [image]
            
        if resolution == None:
            resolution = self.resolution
        
        if self.cache == True:
            self.download_projection_grid_data(eid,
                                               image,
                                               resolution,
                                               path)
        
        data, _ = nrrd.read(path)
        
        return data
            

    # TODO: move these methods to mouse connectivity app
    def cache_injection_density(self, path, eid):
        return self.cache_projection_grid_data(
            path, eid, GridDataApi.INJECTION_DENSITY)


    def cache_projection_density(self, path, eid):
        return self.cache_projection_grid_data(
            path, eid, GridDataApi.PROJECTION_DENSITY)


    def cache_injection_fraction(self, path, eid):
        return self.cache_projection_grid_data(
            path, eid, GridDataApi.INJECTION_FRACTION)


    def cache_data_mask(self, path, eid):
        return self.cache_projection_grid_data(
            path, eid, GridDataApi.DATA_MASK)
        
        
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
        
        Returns
        -------
            file : 3-D expression grid data packaged into a compressed archive file (.zip).
            
        Notes
        -----
        '''
        if include != None:
            include_clause = ''.join(['?include=',
                                      ','.join(include)])
        else:
            include_clause = ''
        
        url = ''.join([self.grid_data_endpoint,
                       '/download/',
                       str(section_data_set_id),
                       include_clause])
        
        if path == None:
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
        
        if image != None:
            params_list.append('image=' +  ','.join(image))
        
        if resolution != None:
            params_list.append('resolution=%d' % (resolution))
        
        if len(params_list) > 0:
            params_clause = '?' + '&'.join(params_list)
        else:
            params_clause = ''
        
        url = ''.join([self.grid_data_endpoint,
                       '/download_file/',
                       str(section_data_set_id),
                       params_clause])
        
        if save_file_path == None:
            save_file_path = str(section_data_set_id) + '.nrrd'
        
        self.retrieve_file_over_http(url, save_file_path)
    
    
    def calculate_centroid(self,
                           injection_density,                           
                           injection_fraction):
        # find all voxels with injection_fraction > 0
        injection_voxels = np.nonzero(injection_fraction)
        injection_density_computed = np.multiply(injection_density[injection_voxels],
                                                 injection_fraction[injection_voxels]) 
        sum_density = np.sum(injection_density_computed)
    
        # compute centroid in CCF coordinates
        if sum_density > 0 :
            centroid = np.dot(injection_density_computed,
                              zip(*injection_voxels)) / sum_density * self.resolution
        else:
            centroid = None
        
        return centroid

