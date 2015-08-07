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

from allensdk.api.queries.rma.rma_simple_api import RmaSimpleApi
from allensdk.api.cache import Cache
import numpy as np
import os, nrrd


class GridDataApi(RmaSimpleApi, Cache):
    '''HTTP Client for the Allen 3-D Expression Grid Data Service.
    
    See: `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data>`_
    '''
    PRODUCT_ID = 5
    
    def __init__(self,
                 resolution=25,
                 base_uri=None,
                 cache=False):
        super(GridDataApi, self).__init__(base_uri)
        Cache.__init__(self, cache=cache)
        self.resolution = resolution

        
    def cache_injection_density(self,
                                path,            
                                eid):
        if self.cache == True:        
            self.download_projection_grid_data(eid,
                                               ['injection_density'],
                                               self.resolution,
                                               path)
            
        injection_density, _ = nrrd.read(path)
        
        return injection_density


    def cache_projection_density(self,
                                 path,
                                 eid):
        if self.cache == True:
            try:
                os.makedirs(os.path.dirname(path))
            except:
                pass
        
            self.download_projection_grid_data(eid,
                                               ['projection_density'],
                                               self.resolution,
                                               path)
            
        projection_density, _ = nrrd.read(path)
        
        return projection_density


    def cache_injection_fraction(self,
                                 path,
                                 eid):
        if self.cache == True:
            try:
                os.makedirs(os.path.dirname(path))
            except:
                pass
    
            self.download_projection_grid_data(eid,
                                               ['injection_fraction'],
                                               self.resolution,
                                               path)

        injection_fraction, _ = nrrd.read(path)
            
        return injection_fraction


    def cache_data_mask(self,
                        path,
                        eid):
        if self.cache == True:
            try:
                os.makedirs(os.path.dirname(path))
            except:
                pass
            
            self.download_projection_grid_data(eid,
                                               ['data_mask'],
                                               self.resolution,
                                               path)
            
        data_mask, _ = nrrd.read(path)
        
        return data_mask
 

    def build_expression_grid_download_query(self,
                                             section_data_set_id,
                                             include=None):
        '''Build the URL.
        
        Parameters
        ----------
        section_data_set_id : integer
            What to download.
        include : list of strings, optional
            Image volumes. 'energy' (default), 'density', 'intensity'. 
        
        Returns
        -------
            string : The url.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        for additional documentation.
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
        
        return url
    
    
    def build_projection_grid_download_query(self,
                                             section_data_set_id,
                                             image=None,
                                             resolution=None):
        '''Build the URL.
        
        Parameters
        ----------
        section_data_set_id : integer
            What to download.
        image : list of strings, optional
            Image volume. 'projection_density', 'projection_energy', 'injection_fraction', 'injection_density', 'injection_energy', 'data_mask'.
        resolution : integer, optional
            in microns. 10, 25, 50, or 100 (default).
        
        Returns
        -------
            string : The url.
        
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
        
        return url
    
    
    def get_experiments(self,
                        product_abbreviation=None,
                        plane_of_section=None,
                        gene_acronym=None,
                        fmt='json'):
        '''Fetch relevant metadata
        including ids for downloading the energy volume
        for an Atlas' experiment.
        
        Parameters
        ----------
        product_abbreviation : string
            i.e. 'Mouse'
        plane_of_section : string, optional
            'coronal' or 'sagittal'
        gene_acronym : string, optional
            i.e. 'Adora2a'
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        and `Example Queries for Experiment Metadata <http://help.brain-map.org/display/api/Example+Queries+for+Experiment+Metadata#ExampleQueriesforExperimentMetadata-MouseBrain>`_
        for additional documentation.
        '''
        criteria = ['[failed$eqfalse]']
        
        if product_abbreviation != None:
            criteria.append(",products[abbreviation$eq'%s']" % 
                            (product_abbreviation))
        
        if plane_of_section != None:
            criteria.append(",plane_of_section[name$eq'%s']" %
                            (plane_of_section))
        
        if gene_acronym != None:
            criteria.append(",genes[acronym$eq'%s']" %
                            (gene_acronym))
        
        result = \
            self.model_query('SectionDataSet',
                             criteria=criteria)
        
        return result
    
    
    def read_response(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
        
    def download_expression_grid_data(self,
                                      section_data_set_id,
                                      include=None,
                                      save_file_path=None):
        '''Download in NRRD format.
        
        Parameters
        ----------
        section_data_set_id : integer
            What to download.
        include : list of strings, optional
            Image volumes. 'energy' (default), 'density', 'intensity'.
        save_file_path : string, optional
            File name to save as.
        
        Returns
        -------
            file : 3-D expression grid data packaged into a compressed archive file (.zip).
            
        Notes
        -----
        '''
        url = self.build_expression_grid_download_query(section_data_set_id,
                                                        include)
        
        if save_file_path == None:
            save_file_path = str(section_data_set_id) + '.zip'
        
        self.retrieve_file_over_http(url, save_file_path)
    
    
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
        url = self.build_projection_grid_download_query(section_data_set_id,
                                                        image,
                                                        resolution)
        
        if save_file_path == None:
            save_file_path = str(section_data_set_id) + '.zip'
        
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

