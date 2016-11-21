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
from six import string_types


class ImageDownloadApi(RmaApi):
    '''HTTP Client to download whole or partial two-dimensional images from the Allen Institute
    with the SectionImage, AtlasImage and ProjectionImage Download Services.

    See `Downloading an Image <http://help.brain-map.org/display/api/Downloading+an+Image>`_
    for more documentation.
    '''

    _FILTER_TYPES = [ 'range', 'rgb', 'contrast' ]
    COLORMAPS = { "gray": 0,
                  "hotmetal": 1,
                  "jet": 2,
                  "redtemp": 3,
                  "expression": 4,
                  "red": 5,
                  "blue": 6,
                  "green": 7,
                  "aba": 8,
                  "aibsmap_alt": 9,
                  "colormap": 10,
                  "projection": 11
    }

    def __init__(self, base_uri=None):
        super(ImageDownloadApi, self).__init__(base_uri)

    def download_section_image(self,
                               section_image_id,
                               file_path=None,
                               **kwargs):
        self.download_image(section_image_id,
                            file_path,
                            endpoint=self.section_image_download_endpoint,
                            **kwargs)

    def download_atlas_image(self,
                             atlas_image_id,
                             file_path=None,
                             **kwargs):
        self.download_image(atlas_image_id,
                            file_path,
                            endpoint=self.atlas_image_download_endpoint,
                            **kwargs)

    def download_projection_image(self,
                                  projection_image_id,
                                  file_path=None,
                                  **kwargs):
        self.download_image(projection_image_id,
                            file_path,
                            endpoint=self.projection_image_download_endpoint,
                            **kwargs)

    def download_image(self,
                       image_id,
                       file_path=None,
                       endpoint=None,
                       **kwargs):
        ''' Download whole or partial two-dimensional images
        from the Allen Institute with the SectionImage or AtlasImage service.

        Parameters
        ----------
        image_id : integer
            SubImage to download.
        file_path : string, optional
            where to put it, defaults to image_id.jpg
        downsample : int, optional
            Number of times to downsample the original image.
        quality : int, optional
            jpeg quality of the returned image, 0 to 100 (default)
        expression : boolean, optional
            Request the expression mask for the SectionImage.
        view : string, optional
            'expression', 'projection', 'tumor_feature_annotation'
            or 'tumor_feature_boundary'
        top : int, optional
            Index of the topmost row of the region of interest.
        left :int, optional
            Index of the leftmost column of the region of interest.
        width : int, optional
            Number of columns in the output image.
        height : int, optional
            Number of rows in the output image.
        range : list of ints, optional
            Filter to specify the RGB channels. low,high,low,high,low,high
        colormap : list of floats, optional
            Filter to specify the RGB channels. [lower_threshold,colormap]
            gain 0-1, colormap id is a string from ImageDownloadApi.COLORMAPS
        rgb : list of floats, optional
            Filter to specify the RGB channels. [red,green,blue] 0-1
        contrast : list of floats, optional
            Filter to specify contrast parameters. [gain,bias] 0-1
        annotation : boolean, optional
            Request the annotated AtlasImage
        atlas : int, optional
            Specify the desired Atlas' annotations.
        projection : boolean, optional
            Request projection for the specified image.
        downsample_dimensions : boolean, optional
            Indicates if the width and height should be adjusted
            to account for downsampling.

        Returns
        -------
        None
            the file is downloaded and saved to the path.

        Notes
        -----
        By default, an unfiltered full-sized image with the highest quality
        is returned as a download if no parameters are provided.

        'downsample=1' halves the number of pixels of the original image
        both horizontally and vertically.        range_list = kwargs.get('range', None)

        
        Specifying 'downsample=2' quarters the height and width values.

        Quality must be an integer from 0, for the lowest quality,
        up to as high as 100. If it is not specified,
        it defaults to the highest quality.

        Top is specified in full-resolution (largest tier) pixel coordinates.
        SectionImage.y is the default value.

        Left is specified in full-resolution (largest tier) pixel coordinates.
        SectionImage.x is the default value.

        Width is specified in tier-resolution (desired tier) pixel coordinates.
        SectionImage.width is the default value. It is automatically adjusted when downsampled.

        Height is specified in tier-resolution (desired tier) pixel coordinates.
        SectionImage.height is the default value. It is automatically adjusted when downsampled.

        The range parameter consists of 6 comma delimited integers
        that define the lower (0) and upper (4095) bound for each channel in red-green-blue order
        (i.e. "range=0,1500,0,1000,0,4095").
        The default range values can be determined by referring to the following fields
        on the Equalization model associated with the SectionDataSet:
        red_lower, red_uppper, green_lower, green_upper, blue_lower, blue_upper.
        For more information, see the
        `Image Controls <http://help.brain-map.org/display/mouseconnectivity/Projection#Projection-ImageControls>`_
        section of the Allen Mouse Brain Connectivity Atlas:
        `Projection Dataset <http://help.brain-map.org/display/mouseconnectivity/Projection>`_
        help topic.
        See: `Image Download Service `<http://help.brain-map.org/display/api/Downloading+an+Image>_
        '''
        params = []

        if endpoint is None:
            endpoint = self.image_download_endpoint

        downsample = kwargs.get('downsample', None)

        if downsample is not None:
            params.append('downsample=%d' % (downsample))

        quality = kwargs.get('quality', None)

        if quality is not None:
            params.append('quality=%d' % (quality))

        tumor_feature_annotation = kwargs.get('tumor_feature_annotation', None)

        if tumor_feature_annotation is not None:
            if tumor_feature_annotation:
                params.append('tumor_feature_annotation=true')
            else:
                params.append('tumor_feature_annotation=false')

        tumor_feature_boundary = kwargs.get('tumor_feature_boundary', None)

        if tumor_feature_boundary is not None:
            if tumor_feature_boundary:
                params.append('tumor_feature_boundary=true')
            else:
                params.append('tumor_feature_boundary=false')

        annotation = kwargs.get('annotation', None)

        if annotation is not None:
            if annotation is True:
                params.append('annotation=true')
            else:
                params.append('annotation=false')

        atlas = kwargs.get('atlas', None)

        if atlas is not None:
            params.append('atlas=%d' % (atlas))

        projection = kwargs.get('projection', None)

        if projection is not None:
            if projection is True:
                params.append('projection=true')
            else:
                params.append('projection=false')

        expression = kwargs.get('expression', None)

        if expression is not None:
            if expression:
                params.append('expression=true')
            else:
                params.append('expression=false')

        colormap_filter = kwargs.get('colormap', None)
        
        if colormap_filter is not None:
            if isinstance(colormap_filter, string_types):
                params.append('colormap=%s' % (colormap_filter))
            else:
                lower_threshold = colormap_filter[0]
                colormap_id = ImageDownloadApi.COLORMAPS[colormap_filter[1]]
                filter_values_list = '0.5,%s,0,256,%d' % (str(lower_threshold),
                                                          colormap_id)
                params.append('colormap=%s' % (filter_values_list))

        # see
        # http://api.brain-map.org/api/v2/data/SectionDataSet/100141599.xml?include=equalization,section_images
        for filter_type in ImageDownloadApi._FILTER_TYPES:
            filter_values = kwargs.get(filter_type, None)
    
            if filter_values is not None:
                filter_values_list = ','.join(str(r) for r in filter_values)
                params.append('%s=%s' % (filter_type, filter_values_list))

        view = kwargs.get('view', None)

        if view is not None:
            if view in ['expression',
                        'projection',
                        'tumor_feature_annotation',
                        'tumor_feature_boundary']:
                params.append('view=%s' % (view))
            else:
                raise ValueError("view argument should be 'expression', 'projection', 'tumor_feature_annotation' or 'tumor_feature_boundary'")

        # region of interest
        for roi_key in ['left', 'top', 'width', 'height']:
            roi_value = kwargs.get(roi_key, None)
            if roi_value is not None:
                params.append('%s=%d' % (roi_key, roi_value))

        downsample_dimensions = kwargs.get('downsample_dimensions', None)

        if downsample_dimensions is not None:
            if downsample_dimensions:
                params.append('downsample_dimensions=true')
            else:
                params.append('downsample_dimensions=false')

        if len(params) > 0:
            url_params = "?" + "&".join(params)
        else:
            url_params = ''

        image_url = ''.join([endpoint,
                             '/',
                             str(image_id),
                             url_params])

        if file_path is None:
            file_path = '%d.jpg' % (image_id)

        self.retrieve_file_over_http(image_url, file_path)


    def atlas_image_query(self, atlas_id, image_type_name=None):
        '''Build the URL.

        Parameters
        ----------
        atlas_id : integer, optional
            request a certain record.
        image_type_name : string, optional
            if not present, the query will get it from the atlas id.

        Returns
        -------
        url : string
            The constructed URL

        Notes
        -----
        See `Downloading Atlas Images and Graphics <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-DownloadingAtlasImagesAndGraphics>`_
        for additional documentation.
        The atlas id can be found with :py:meth:`allensdk.api.queries.ontologies_api.OntologiesApi.build_atlases_query`
        '''
        rma = self

        stages = []

        if image_type_name is None:
            atlas_stage = rma.model_stage('Atlas',
                                          criteria='[id$eq%d]' % (atlas_id),
                                          only=['image_type'])
            stages.append(atlas_stage)

            atlas_name_pipe_stage = rma.pipe_stage('list',
                                                   parameters=[('type_name',
                                                                rma.IS,
                                                                rma.quote_string('image_type'))])
            stages.append(atlas_name_pipe_stage)

            image_type_name = '$type_name'
        else:
            image_type_name = rma.quote_string(image_type_name)

        criteria_list = ['[annotated$eqtrue],',
                         'atlas_data_set(atlases[id$eq%d]),' % (atlas_id),
                         "alternate_images[image_type$eq%s]" % (image_type_name)]

        atlas_image_model_stage = rma.model_stage('AtlasImage',
                                                  criteria=criteria_list,
                                                  order=[
                                                      'sub_images.section_number'],
                                                  num_rows='all')

        stages.append(atlas_image_model_stage)

        return self.json_msg_query(
            rma.build_query_url(stages))
