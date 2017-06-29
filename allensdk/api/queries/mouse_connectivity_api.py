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
from .grid_data_api import GridDataApi
from ..cache import cacheable, Cache
import numpy as np
import nrrd
import six


class MouseConnectivityApi(RmaApi):
    '''
    HTTP Client for the Allen Mouse Brain Connectivity Atlas.

    See: `Mouse Connectivity API <http://help.brain-map.org/display/mouseconnectivity/API>`_
    '''
    PRODUCT_IDS = [5, 31]

    def __init__(self, base_uri=None):
        super(MouseConnectivityApi, self).__init__(base_uri)

    AVERAGE_TEMPLATE = 'average_template'
    ARA_NISSL = 'ara_nissl'
    MOUSE_2011 = 'annotation/mouse_2011'
    DEVMOUSE_2012 = 'annotation/devmouse_2012'
    CCF_2015 = 'annotation/ccf_2015'
    CCF_2016 = 'annotation/ccf_2016'
    CCF_VERSION_DEFAULT = CCF_2016

    VOXEL_RESOLUTION_10_MICRONS = 10
    VOXEL_RESOLUTION_25_MICRONS = 25
    VOXEL_RESOLUTION_50_MICRONS = 50
    VOXEL_RESOLUTION_100_MICRONS = 100

    @cacheable(strategy='create',
               reader=nrrd.read,
               pathfinder=Cache.pathfinder(file_name_position=3,
                                           path_keyword='file_name'))
    def download_annotation_volume(self,
                                   ccf_version,
                                   resolution,
                                   file_name):
        '''
        Download the annotation volume at a particular resolution.

        Parameters
        ----------
        ccf_version: string
            MouseConnectivityApi.CCF_2016 (default) or MouseConnectivityApi.CCF_2015
        resolution: int
            Desired resolution to download in microns.
            Must be 10, 25, 50, or 100.
        file_name: string
            Where to save the annotation volume.
        
        Note: the parameters must be used as positional parameters, not keywords
        '''

        if ccf_version is None:
            ccf_version = MouseConnectivityApi.CCF_VERSION_DEFAULT

        self.download_volumetric_data(ccf_version,
                                      'annotation_%d.nrrd' % resolution, 
                                      save_file_path=file_name)

    @cacheable(strategy='create',
               reader=nrrd.read,
               pathfinder=Cache.pathfinder(file_name_position=2,
                                           path_keyword='file_name'))
    def download_template_volume(self, resolution, file_name):
        '''
        Download the registration template volume at a particular resolution.

        Parameters
        ----------

        resolution: int
            Desired resolution to download in microns.  Must be 10, 25, 50, or 100.

        file_name: string
            Where to save the registration template volume.
        '''
        self.download_volumetric_data(MouseConnectivityApi.AVERAGE_TEMPLATE,
                                      'average_template_%d.nrrd' % resolution, 
                                      save_file_path=file_name)


    @cacheable()
    def get_experiments(self,
                        structure_ids,
                        **kwargs):
        '''
        Fetch experiment metadata from the Mouse Brain Connectivity Atlas.

        Parameters
        ----------
        structure_ids : integer or list, optional
            injection structure

        Returns
        -------
        url : string
            The constructed URL
        '''
        criteria_list = ['[failed$eqfalse]',
                         'products[id$in%s]' % (','.join(str(i) for i in MouseConnectivityApi.PRODUCT_IDS))]

        if structure_ids is not None:
            if type(structure_ids) is not list:
                structure_ids = [structure_ids]
            criteria_list.append('[id$in%s]' % ','.join(str(i)
                                                        for i in structure_ids))

        criteria_string = ','.join(criteria_list)

        return self.model_query('SectionDataSet',
                                criteria=criteria_string,
                                **kwargs)

    @cacheable()
    def get_manual_injection_summary(self, experiment_id):
        ''' Retrieve manual injection summary. '''

        criteria = '[id$in%d]' % (experiment_id)

        include = ['specimen(donor(transgenic_mouse(transgenic_lines)),',
                   'injections(structure,age)),',
                   'equalization,products']

        only = ['id',
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
                'donors.id']

        return self.model_query('SectionDataSet',
                                criteria=criteria,
                                include=include,
                                only=only)

    @cacheable()
    def get_experiment_detail(self, experiment_id):
        '''Retrieve the experiments data.'''

        criteria = '[id$eq%d]' % (experiment_id)
        include = ['specimen(stereotaxic_injections(primary_injection_structure,structures,stereotaxic_injection_coordinates)),',
                   'equalization,',
                   'sub_images']
        order = ["'sub_images.section_number$asc'"]

        return self.model_query('SectionDataSet',
                                criteria=criteria,
                                include=include,
                                order=order)

    @cacheable()
    def get_projection_image_info(self,
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
        Download the image using :py:meth:`allensdk.api.queries.image_download_api.ImageDownloadApi.download_section_image`
        '''

        criteria = '[id$eq%d]' % (experiment_id)
        include = ['equalization,sub_images[section_number$eq%d]' %
                   (section_number)]

        return self.model_query('SectionDataSet',
                                criteria=criteria,
                                include=include)

    def build_volumetric_data_download_url(self,
                                           data_path,
                                           file_name,
                                           voxel_resolution=None,
                                           release=None,
                                           coordinate_framework=None):
        '''Construct url to download 3D reference model in NRRD format.

        Parameters
        ----------
        data_path : string
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

        if voxel_resolution is None:
            voxel_resolution = MouseConnectivityApi.VOXEL_RESOLUTION_10_MICRONS

        if release is None:
            release = 'current-release'

        if coordinate_framework is None:
            coordinate_framework = 'mouse_ccf'

        url = ''.join([self.informatics_archive_endpoint,
                       '/%s/%s/' % (release, coordinate_framework),
                       data_path,
                       '/',
                       file_name])

        return url


    def download_volumetric_data(self,
                                 data_path,
                                 file_name,
                                 voxel_resolution=None,
                                 save_file_path=None,
                                 release=None,
                                 coordinate_framework=None):
        '''Download 3D reference model in NRRD format.

        Parameters
        ----------
        data_path : string
            'average_template', 'ara_nissl', 'annotation/ccf_2015', 'annotation/mouse_2011', or 'annotation/devmouse_2012'
        file_name : string
            server-side file name. 'annotation_10.nrrd' for example.
        voxel_resolution : int
            10, 25, 50 or 100
        coordinate_framework : string
            'mouse_ccf' (default) or 'mouse_annotation'

        Notes
        -----
        See: `3-D Reference Models <http://help.brain-map.org/display/mouseconnectivity/API#API-3DReferenceModels>`_
        for additional documentation.
        '''
        url = self.build_volumetric_data_download_url(data_path,
                                                      file_name,
                                                      voxel_resolution,
                                                      release,
                                                      coordinate_framework)

        if save_file_path is None:
            save_file_path = file_name

        if save_file_path is None:
            save_file_path = 'volumetric_data.nrrd'

        self.retrieve_file_over_http(url, save_file_path)
        

    def download_reference_aligned_image_channel_volumes(self,
                                                         data_set_id,
                                                         save_file_path=None):
        '''
        Returns
        -------
            The well known file is downloaded
        '''
        well_known_file_url = self.get_reference_aligned_image_channel_volumes_url(
            data_set_id)

        if save_file_path is None:
            save_file_path = str(data_set_id) + '.zip'

        self.retrieve_file_over_http(well_known_file_url, save_file_path)

    def build_reference_aligned_image_channel_volumes_url(self,
                                                          data_set_id):
        '''Construct url to download the red, green, and blue channels
        aligned to the 25um adult mouse brain reference space volume.

        Parameters
        ----------
        data_set_id : integerallensdk.api.queries
            aka attachable_id

        Notes
        -----
        See: `Reference-aligned Image Channel Volumes <http://help.brain-map.org/display/mouseconnectivity/API#API-ReferencealignedImageChannelVolumes>`_
        for additional documentation.
        '''

        criteria = ['well_known_file_type',
                    "[name$eq'ImagesResampledTo25MicronARA']",
                    "[attachable_id$eq%d]" % (data_set_id)]

        model_stage = self.model_stage('WellKnownFile',
                                       criteria=criteria)

        url = self.build_query_url([model_stage])

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
                                      lambda parsed_json: str(
                                          parsed_json['msg'][0]['download_link']),
                                      data_set_id)

        url = self.api_url + download_link

        return url

    def experiment_source_search(self, **kwargs):
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
        product_ids : list of integers, optional
            Integer Product.id
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
        tuples = [(k, v) for k, v in six.iteritems(kwargs)]
        return self.service_query('mouse_connectivity_injection_structure', parameters=tuples)

    def experiment_spatial_search(self, **kwargs):
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
        product_ids : list of integers, optional
            Integer Product.id
        start_row : integer, optional
            For paging purposes. Defaults to 0.
        num_rows : integer, optional
            For paging purposes. Defaults to 2000.

        Notes
        -----
        See `Spatial Search <http://help.brain-map.org/display/mouseconnectivity/API#API-SpatialSearch>`_
        and
        `service::mouse_connectivity_target_spatial <http://help.brain-map.org/display/api/Connected+Services+and+Pipes#ConnectedServicesandPipes-service%3A%3Amouseconnectivitytargetspatial>`_.

        '''

        tuples = [(k, v) for k, v in six.iteritems(kwargs)]
        return self.service_query('mouse_connectivity_target_spatial', parameters=tuples)

    def experiment_injection_coordinate_search(self, **kwargs):
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
        product_ids : list of integers, optional
            Integer Product.id
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
        tuples = [(k, v) for k, v in six.iteritems(kwargs)]
        return self.service_query('mouse_connectivity_injection_coordinate', parameters=tuples)

    def experiment_correlation_search(self, **kwargs):
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
        product_ids : list of integers, optional
            Integer Product.id
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
        tuples = sorted(six.iteritems(kwargs))
        return self.service_query('mouse_connectivity_correlation',
                                  parameters=tuples)

    @cacheable()
    def get_structure_unionizes(self,
                                experiment_ids,
                                is_injection=None,
                                structure_name=None,
                                structure_ids=None,
                                hemisphere_ids=None,
                                normalized_projection_volume_limit=None,
                                include=None,
                                debug=None,
                                order=None):

        experiment_filter = '[section_data_set_id$in%s]' %\
                            ','.join(str(i) for i in experiment_ids)

        if is_injection is True:
            is_injection_filter = '[is_injection$eqtrue]'
        elif is_injection is False:
            is_injection_filter = '[is_injection$eqfalse]'
        else:
            is_injection_filter = ''

        if normalized_projection_volume_limit is not None:
            volume_filter = '[normalized_projection_volume$gt%f]' %\
                            (normalized_projection_volume_limit)
        else:
            volume_filter = ''

        if hemisphere_ids is not None:
            hemisphere_filter = '[hemisphere_id$in%s]' %\
                ','.join(str(h) for h in hemisphere_ids)
        else:
            hemisphere_filter = ''

        if structure_name is not None:
            structure_filter = ",structure[name$eq'%s']" % (structure_name)
        elif structure_ids is not None:
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

    @cacheable(strategy='create', 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_injection_density(self, path, experiment_id, resolution):
        GridDataApi(base_uri=self.api_url).download_projection_grid_data(
            experiment_id, [GridDataApi.INJECTION_DENSITY], resolution, path)

    @cacheable(strategy='create', 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_projection_density(self, path, experiment_id, resolution):
        GridDataApi(base_uri=self.api_url).download_projection_grid_data(
            experiment_id, [GridDataApi.PROJECTION_DENSITY], resolution, path)

    @cacheable(strategy='create', 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_injection_fraction(self, path, experiment_id, resolution):
        GridDataApi(base_uri=self.api_url).download_projection_grid_data(
            experiment_id, [GridDataApi.INJECTION_FRACTION], resolution, path)

    @cacheable(strategy='create', 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_data_mask(self, path, experiment_id, resolution):
        GridDataApi(base_uri=self.api_url).download_projection_grid_data(
            experiment_id, [GridDataApi.DATA_MASK], resolution, path)

    def calculate_injection_centroid(self,
                                     injection_density,
                                     injection_fraction,
                                     resolution=25):
        '''
        Compute the centroid of an injection site.

        Parameters
        ----------

        injection_density: np.ndarray
            The injection density volume of an experiment

        injection_fraction: np.ndarray
            The injection fraction volume of an experiment

        '''

        # find all voxels with injection_fraction > 0
        injection_voxels = np.nonzero(injection_fraction)
        injection_density_computed = np.multiply(injection_density[injection_voxels],
                                                 injection_fraction[injection_voxels])
        sum_density = np.sum(injection_density_computed)

        # compute centroid in CCF coordinates
        if sum_density > 0:
            centroid = np.dot(injection_density_computed,
                              list(zip(*injection_voxels))) / sum_density * resolution
        else:
            centroid = None

        return centroid
