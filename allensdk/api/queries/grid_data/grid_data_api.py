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
from allensdk.api.queries.rma.rma_api import RmaApi

class GridDataApi(Api):
    '''HTTP Client for the Allen 3-D Expression Grid Data Service.
    
    See: `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data>`_
    '''
    PRODUCT_ID = 5
    
    def __init__(self, base_uri=None):
        super(GridDataApi, self).__init__(base_uri)
    
    
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
    
    
    def build_experiment_id_url(self,
                                product_abbreviation=None,
                                plane_of_section=None,
                                gene_acronym=None,
                                fmt='json'):
        '''Build the URL to get relevant experiment ids
        for downloading the energy volume
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
        rma = RmaApi()
        
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
        
        section_data_set_stage = \
            rma.model_stage('SectionDataSet',
                            criteria=criteria)
        
        return rma.build_query_url(section_data_set_stage)
    
    
    def read_response(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_experiment_ids(self,
                           product_abbreviation=None,
                           plane_of_section=None,
                           gene_acronym=None):
        '''Build the URL to get relevant experiment ids
        for downloading the energy volume
        for an Atlas' experiment.
        
        Parameters
        ----------
        product_abbreviation : string
            i.e. 'Mouse'
        plane_of_section : string, optional
            'coronal' or 'sagittal'
        gene_acronym : string, optional
            i.e. 'Adora2a'
        
        Returns
        -------
        dict : the parsed json response containing experiment records.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        and `Example Queries for Experiment Metadata <http://help.brain-map.org/display/api/Example+Queries+for+Experiment+Metadata#ExampleQueriesforExperimentMetadata-MouseBrain>`_
        for additional documentation.
        '''
        return self.do_query(self.build_experiment_id_url,
                             self.read_response,
                             product_abbreviation,
                             plane_of_section,
                             gene_acronym)
    
    
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
