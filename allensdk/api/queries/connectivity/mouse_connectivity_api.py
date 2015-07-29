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
from allensdk.api.queries.rma.connected_services import ConnectedServices


class MouseConnectivityApi(RmaSimpleApi):
    '''HTTP Client for the Allen Mouse Brain Connectivity Atlas.
    
    See: `Mouse Connectivity API <http://help.brain-map.org/display/mouseconnectivity/API>`_
    '''
    PRODUCT_ID = 5
    
    def __init__(self, base_uri=None):
        super(MouseConnectivityApi, self).__init__(base_uri)
    
    
    def build_query(self, structure_id=None, fmt='json'):
        '''Build the URL that will fetch experiments 
        in the "Mouse Connectivity Projection" Product.
        
        Parameters
        ----------
        structure_id : integer, optional
            injection structure
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        
        if structure_id:
            structure_filter = '[id$eq%d]' % (structure_id)
        else:
            structure_filter = ''
        
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       '[failed$eqfalse],'
                       'products[id$eq%d]' % (MouseConnectivityApi.PRODUCT_ID),
                       ',rma::include,',
                       'specimen',
                       '(stereotaxic_injections',
                       '(primary_injection_structure,',
                       'structures',
                       structure_filter,
                       '))'])
        
        return url
    
    
    def build_manual_injection_summary_url(self, experiment_id, fmt='json'):
        '''Construct a query for summary table for one experiment.
        
        Parameters
        ----------
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
         Notes
         -----
         Based on the connectivity application detail page.
        '''
        rma = RmaSimpleApi()
        model_stage = \
            rma.model_stage(
                model='SectionDataSet',
                criteria='[id$in%d]' % (experiment_id),
                include=['specimen(donor(transgenic_mouse(transgenic_lines)),',
                         'injections(structure,age)),',
                         'equalization,products'],
                only=['id',
                      'failed',
                      'storage_directory',
                      'red_lower',
                      'red_upper',
                      'green_lower',
                      'green_upper',
                      'blue_lower',
                      'blue_upper',
                      'products.id',
                      'specimen_id',
                      'structure_id',
                      'reference_space_id',
                      'primary_injection_structure_id',
                      'registration_point',
                      'coordinates_ap',
                      'coordinates_dv',
                      'coordinates_ml',
                      'angle',
                      'sex',
                      'strain',
                      'injection_materials',
                      'acronym',
                      'structures.name',
                      'days',
                      'transgenic_mice.name',
                      'transgenic_lines.name',
                      'transgenic_lines.description',
                      'transgenic_lines.id',
                      'donors.id'])
    
        return rma.build_query_url(model_stage)
    
    
    def build_detail_query(self, experiment_id, fmt='json'):
        '''Construct a query for detailed metadata for one experiment.
        
        Parameters
        ----------
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       '[id$eq%d]' % (experiment_id),
                       ',rma::include,',
                       'specimen',
                       '(stereotaxic_injections',
                       '(primary_injection_structure,',
                       'structures,',
                       'stereotaxic_injection_coordinates)),',
                       'equalization,',
                       'sub_images',
                       ',rma::options',
                       "[order$eq'sub_images.section_number$asc']"])
        
        return url
    
    
    def build_projection_image_meta_info(self,
                                         experiment_id,
                                         section_number,
                                         fmt='json'):
        '''Construct URL to fetch meta-information of one projection image.
        
        Parameters
        ----------
        experiment_id : integer
        section_number : integer
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::SectionDataSet',
                       ',rma::criteria,',
                       '[id$eq%d]' % (experiment_id),
                       ',rma::include,',
                       'equalization,',
                       'sub_images',
                       '[section_number$eq%d]' % (section_number)])
        
        return url
    
    # TODO: deprecate for fetch_volume
    def build_structure_projection_signal_statistics_url(self,
                                                         section_data_set_id,
                                                         is_injection=None,
                                                         fmt='json'):
        '''Query for projection signal statistics for one injection experiment.
        
        Parameters
        ----------
        section_data_set_id : integer
        is_injection : boolean, optional
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
        Notes
        -----
        See: examples under `Projection Structure Unionization <http://help.brain-map.org/display/mouseconnectivity/API#API-ProjectionStructureUnionization>`_.
        
        '''
        criteria_params = [',rma::criteria,',
                           '[section_data_set_id$eq%d]' % (section_data_set_id)]
        
        if is_injection != None:
            if is_injection:
                criteria_params.append('[is_injection$eqtrue]')
            else:
                criteria_params.append('[is_injection$eqtrue]')
        
        criteria_clause = ''.join(criteria_params)
        include_clause = ''.join([',rma::include,',
                                  'structure'])
        options_clause = ''.join([',rma::options,',
                                  "[num_rows$eq'all']"])
        
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::ProjectionStructureUnionize',
                       criteria_clause,
                       include_clause,
                       options_clause])
        
        return url
    
    
    def build_signal_statistics_url(self,
                                    section_data_set_id,
                                    is_injection=None,
                                    fmt='json'):
        '''Query for projection signal statistics for one injection experiment.
        
        Parameters
        ----------
        section_data_set_id : integer
        is_injection : boolean, optional
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
        Notes
        -----
        See: examples under `Projection Structure Unionization <http://help.brain-map.org/display/mouseconnectivity/API#API-ProjectionStructureUnionization>`_.
        Setting is_injection to False will get the projection signal statistics exclusive of injection area.
        Setting is_injection to True will get the injection site statistics for the experiment.
        '''
        criteria_params = [',rma::criteria,',
                           '[section_data_set_id$eq%d]' % (section_data_set_id)]
        
        if is_injection != None:
            if is_injection:
                criteria_params.append('[is_injection$eqtrue]')
            else:
                criteria_params.append('[is_injection$eqtrue]')
        
        criteria_clause = ''.join(criteria_params)
        include_clause = ''.join([',rma::include,',
                                  'structure'])
        options_clause = ''.join([',rma::options,',
                                  '[num_rows$eq5000]'])
        
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       'model::ProjectionStructureUnionize',
                       criteria_clause,
                       include_clause,
                       options_clause])
        
        return url
    
    
    def build_projection_grid_search_url(self, **kwargs):
        '''Search over the whole projection signal statistics dataset
        to find experiments with specific projection profiles.
        
        Parameters
        ----------
        injection_structures : list of integers or strings
            Integer Structure.id or String Structure.acronym.
        target_domain : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        injection_hemisphere : string, optional
            'right' or 'left', Defaults to both hemispheres.
        target_hemisphere : string, optional
            'right' or 'left', Defaults to both hemispheres.
        transgenic_lines : list of integers or strings, optional
             Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
        injection_domain : list of integers or strings, optional
             Integer Structure.id or String Structure.acronym.
        primary_structure_only : boolean, optional
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.
        
        Notes
        -----
        See `Source Search <http://help.brain-map.org/display/mouseconnectivity/API#API-SourceSearch>`_,
        `Target Search <http://help.brain-map.org/display/mouseconnectivity/API#API-TargetSearch>`_,
        and 
        `service::mouse_connectivity_injection_structure <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivityinjectionstructure>`_.
        
        '''
        svc = ConnectedServices()
        
        service_name = 'mouse_connectivity_injection_structure'
        url = svc.build_url(service_name, kwargs)
        
        return url
    
    
    def build_projection_grid_spatial_search_url(self, **kwargs):
        '''Displays all SectionDataSets
        with projection signal density >= 0.1 at the seed point.
        This service also returns the path
        along the most dense pixels from the seed point
        to the center of each injection site..
        
        Parameters
        ----------
        seed_point : list of floats
            The coordinates of a point in 3-D SectionDataSet space.
        transgenic_lines : list of integers or strings, optional
            Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
        section_data_sets : list of integers, optional
            Ids to filter the results.
        injection_structures : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        primary_structure_only : boolean, optional
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.
        
        Notes
        -----
        See `Spatial Search <http://help.brain-map.org/display/mouseconnectivity/API#API-SpatialSearch>`_
        and 
        `service::mouse_connectivity_injection_structure <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivitytargetspatial>`_.
        
        '''
        svc = ConnectedServices()
        
        service_name = 'mouse_connectivity_target_spatial'
        url = svc.build_url(service_name, kwargs)
        
        return url
    
    
    def build_projection_grid_injection_coordinate_search_url(self, **kwargs):
        '''User specifies a seed location within the 3D reference space.
        The service returns a rank list of experiments
        by distance of its injection site to the specified seed location.
        
        Parameters
        ----------
        seed_point : list of floats
            The coordinates of a point in 3-D SectionDataSet space.
        transgenic_lines : list of integers or strings, optional
            Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
        injection_structures : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        primary_structure_only : boolean, optional
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.
        
        Notes
        -----
        See `Injection Coordinate Search <http://help.brain-map.org/display/mouseconnectivity/API#API-InjectionCoordinateSearch>`_
        and 
        `service::mouse_connectivity_injection_coordinate <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivityinjectioncoordinate>`_.
        
        '''
        svc = ConnectedServices()
        
        service_name = 'mouse_connectivity_injection_coordinate'
        url = svc.build_url(service_name, kwargs)
        
        return url
    
    
    def build_projection_grid_correlation_search_url(self, **kwargs):
        '''Select a seed experiment and a domain over
        which the similarity comparison is to be made.
        
        
        Parameters
        ----------
        row : integer
            SectionDataSet.id to correlate against.
        structures : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        hemisphere : string, optional
            Use 'right' or 'left'. Defaults to both hemispheres.
        transgenic_lines : list of integers or strings, optional
            Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
        injection_structures : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        primary_structure_only : boolean, optional
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.
        
        Notes
        -----
        See `Correlation Search <http://help.brain-map.org/display/mouseconnectivity/API#API-CorrelationSearch>`_
        and 
        `service::mouse_connectivity_correlation <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivitycorrelation>`_.
        
        '''
        svc = ConnectedServices()
        
        service_name = 'mouse_connectivity_correlation'
        url = svc.build_url(service_name, kwargs)
        
        return url
    
    
    def read_response(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_experiments(self, structure_id):
        '''Retrieve the experimants data.'''
        data = self.do_query(self.build_query,
                             self.read_response,
                             structure_id)
        
        return data
    
    
    def get_manual_injection_summary(self, experiment_id):
        '''Retrieve manual injection summary.'''
        data = self.do_query(self.build_manual_injection_summary_url,
                             self.read_response,
                             experiment_id)
        
        return data
    
    
    def get_experiment_detail(self, experiment_id):
        '''Retrieve the experiments data.'''
        data = self.do_query(self.build_detail_query,
                             self.read_response,
                             experiment_id)
        
        return data
    
    
    def get_projection_image_meta_info(self,
                                       experiment_id,
                                       section_number):
        '''Fetch meta-information of one projection image.
        
        Parameters
        ----------
        experiment_id : integer
        
        section_number : integer
        
        Notes
        -----
        See: image examples under 
        `Experimental Overview and Metadata <http://help.brain-map.org/display/mouseconnectivity/API##API-ExperimentalOverviewandMetadata>`_
        for additional documentation.
        Download the image using :py:meth:`allensdk.api.queries.image.image_download_api.ImageDownloadApi.download_section_image`
        '''
        data = self.do_query(self.build_projection_image_meta_info,
                             self.read_response,
                             experiment_id,
                             section_number)
        
        return data
    
    
    def get_structure_projection_signal_statistics(self,
                                                   section_data_set_id,
                                                   is_injection=None):
        '''Fetch meta-information of one projection image.
        
        Parameters
        ----------
        experiment_id : integer
        section_number : integer
        
        Returns
        -------
        data : dict
            Parsed JSON data message.
        
        Notes
        -----
        See: examples under `Projection Structure Unionization <http://help.brain-map.org/display/mouseconnectivity/API#API-ProjectionStructureUnionization>`_.
        '''
        data = self.do_query(self.build_signal_statistics_url,
                             self.read_response,
                             section_data_set_id,
                             is_injection)
        
        return data
    
    
    def build_volumetric_data_download_url(self,
                                           data,
                                           file_name,
                                           voxel_resolution=None,
                                           release=None,
                                           coordinate_framework=None):
        '''Construct url to download 3D reference model in NRRD format.
        
        Parameters
        ----------
        data : string
            'average_template', 'ara_nissl', 'annotation/ccf_2015', 'annotation/mouse_2011', or 'annotation/devmouse_2012'
        voxel_resolution : int
            10, 25, 50 or 100
        coordinate_framework : string
            'mouse_ccf' (default) or 'mouse_annotation'
            
        Notes
        -----
        See: `3-D Reference Models <http://help.brain-map.org/display/mouseconnectivity/API#API-3DReferenceModels>`_ 
        for additional documentation.
        '''
        
        if voxel_resolution == None:
            voxel_resolution = 10
            
        if release == None:
            release = 'current-release'
        
        if coordinate_framework == None:
            coordinate_framework = 'mouse_ccf'
        
        url = ''.join([self.informatics_archive_endpoint,
                       '/%s/%s/' % (release, coordinate_framework),
                       data,
                       '/',
                       file_name])
        
        return url
    
    
    def read_reference_aligned_image_channel_volumes_response(self, parsed_json):
        return parsed_json['msg']        
    
    
    def build_reference_aligned_image_channel_volumes_url(self,
                                                          data_set_id):
        '''Construct url to download the red, green, and blue channels
        aligned to the 25um adult mouse brain reference space volume.
        
        Parameters
        ----------
        data_set_id : integer
            aka attachable_id
            
        Notes
        -----
        See: `Reference-aligned Image Channel Volumes <http://help.brain-map.org/display/mouseconnectivity/API#API-ReferencealignedImageChannelVolumes>`_ 
        for additional documentation.
        '''
        rma = RmaSimpleApi()
        
        criteria = ['well_known_file_type',
                    "[name$eq'ImagesResampledTo25MicronARA']",
                    "[attachable_id$eq%d]" % (data_set_id)]
        
        model_stage = rma.model_stage('WellKnownFile',
                                      criteria=criteria)
        
        url = rma.build_query_url([model_stage])
        
        return url
    
    
    def get_reference_aligned_image_channel_volumes_url(self,
                                                        data_set_id):
        '''Retrieve the download link for a specific data set.\
        
        Notes
        -----
        See `Reference-aligned Image Channel Volumes <http://help.brain-map.org/display/mouseconnectivity/API#API-ReferencealignedImageChannelVolumes>`_
        for additional documentation.
        '''
        download_link = self.do_query(self.build_reference_aligned_image_channel_volumes_url,
                                      lambda parsed_json: str(parsed_json['msg'][0]['download_link']),
                                      data_set_id)
        
        url = self.api_url + download_link
        
        return url
    
    
    def download_volumetric_data(self,
                                 data,
                                 file_name,
                                 voxel_resolution=None,
                                 save_file_path=None,
                                 release=None,
                                 coordinate_framework=None):
        '''Download 3D reference model in NRRD format.
        
        Parameters
        ----------
        data : string
            'average_template', 'ara_nissl', 'annotation/ccf_2015', 'annotation/mouse_2011', or 'annotation/devmouse_2012'
        file_name : string
            
        voxel_resolution : int
            10, 25, 50 or 100
        coordinate_framework : string
            'mouse_ccf' (default) or 'mouse_annotation'
            
        Notes
        -----
        See: `3-D Reference Models <http://help.brain-map.org/display/mouseconnectivity/API#API-3DReferenceModels>`_ 
        for additional documentation.
        '''
        url = self.build_volumetric_data_download_url(data,
                                                      file_name,
                                                      voxel_resolution,
                                                      release,
                                                      coordinate_framework)
        
        self.retrieve_file_over_http(url, save_file_path)
    
    
    def download_reference_aligned_image_channel_volumes_url(self,
                                                             data_set_id,
                                                             save_file_path=None):
        '''
        Returns
        -------
            The well known file is downloaded
        '''
        well_known_file_url = self.get_reference_aligned_image_channel_volumes_url(data_set_id)
        
        if save_file_path == None:
            save_file_path = str(data_set_id) + '.zip'
            
        self.retrieve_file_over_http(well_known_file_url, save_file_path)
    
    
    def get_projection_grid(self, **kwargs):
        '''Search over the whole projection signal statistics dataset
        to find experiments with specific projection profiles.
        
        Parameters
        ----------
        injection_structures : list of integers or strings
            Integer Structure.id or String Structure.acronym.
        target_domain : list of integers or strings, optional
            Integer Structure.id or String Structure.acronym.
        injection_hemisphere : string, optional
            'right' or 'left', Defaults to both hemispheres.
        target_hemisphere : string, optional
            'right' or 'left', Defaults to both hemispheres.
        transgenic_lines : list of integers or strings, optional
             Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
        injection_domain : list of integers or strings, optional
             Integer Structure.id or String Structure.acronym.
        primary_structure_only : boolean, optional
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.
        
        Notes
        -----
        See `Source Search <http://help.brain-map.org/display/mouseconnectivity/API#API-SourceSearch>`_,
        `Target Search <http://help.brain-map.org/display/mouseconnectivity/API#API-TargetSearch>`_,
        and 
        `service::mouse_connectivity_injection_structure <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivityinjectionstructure>`_.
        
        '''
        data = self.do_query(self.build_projection_grid_search_url,
                             self.read_response,
                             **kwargs)
        
        return data
    
    
    def get_projection_grid_spatial(self, **kwargs):
        data = self.do_query(self.build_projection_grid_spatial_search_url,
                             self.read_response,
                             **kwargs)
        
        return data
    
    
    def get_projection_grid_injection_coordinate(self, **kwargs):
        data = self.do_query(self.build_projection_grid_injection_coordinate_search_url,
                             self.read_response,
                             **kwargs)
        
        return data
    
    
    def get_projection_grid_injection_correlation(self, **kwargs):
        data = self.do_query(self.build_projection_grid_injection_correlation_search_url,
                             self.read_response,
                             **kwargs)
        
        return data
    
    def fetch_volume(self,
                     experiment_ids,
                     is_injection,
                     structure_name=None,
                     structure_ids=None,
                     hemisphere_ids=None,
                     normalized_projection_volume_limit=None,
                     include=None,
                     debug=None,
                     order=None):
        experiment_filter = '[section_data_set_id$in%s]' %\
                            ','.join(str(i) for i in experiment_ids)
        
        if is_injection == True:
            is_injection_filter = '[is_injection$eqtrue]'
        elif is_injection == False:
            is_injection_filter = '[is_injection$eqfalse]'
        else:
            is_injection_filter = ''
        
        if normalized_projection_volume_limit != None:
            volume_filter = '[normalized_projection_volume$gt%f]' %\
                            (normalized_projection_volume_limit)
        else:
            volume_filter = ''
        
        if hemisphere_ids == None:
            hemisphere_ids = [3] # both
        
        hemisphere_filter = '[hemisphere_id$in%s]' %\
                            ','.join(str(h) for h in hemisphere_ids)
        
        if structure_name != None:
            structure_filter = ",structure[name$eq'%s']" % (structure_name)
        elif structure_ids != None:
            structure_filter = '[structure_id$in%s]' %\
                               ','.join(str(i) for i in structure_ids)
        else:
            structure_filter = ''
        
        return self.model_query(
                'ProjectionStructureUnionize',
                criteria=''.join([experiment_filter,
                                  is_injection_filter,
                                  volume_filter,
                                  hemisphere_filter,
                                  structure_filter]),
                include=include,
                order=order,
                num_rows='all',
                debug=debug,
                count=False)