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

class ImageDownloadApi(Api):
    '''HTTP Client to download whole or partial two-dimensional images from the Allen Institute
    with the SectionImage, AtlasImage and ProjectionImage Download Services.
    
    See `Downloading and Image <http://help.brain-map.org/display/api/Downloading+an+Image>`_
    for more documentation.
    '''
    def __init__(self, base_uri=None):
        super(ImageDownloadApi, self).__init__(base_uri)
    
    
    def build_section_image_url(self,
                                section_image_id,
                                **kwargs):
        '''
        
        Parameters
        ----------
        section_image_id : integer
            Image to download.
        downsample : int, optional
            Number of times to downsample the original image.
        quality : int, optional
            jpeg quality of the returned image, 0 to 100 (default)
        expression : boolean, optional
            True to retrieve the specified SectionImage expression mask image.
        top : int, optional
            Index of the topmost row of the region of interest.
        left :int, optional
            Index of the leftmost column of the region of interest.
        width : int, optional
            Number of columns in the output image.
        height : int, optional
            Number of rows in the output image.
        range : list of ints, optional
            Filter to specify the RGB channels.
        tumor_feature_annotation : boolean, optional
            True to retrieve the color block image for a Glioblastoma SectionImage.
        tumor_feature_boundary : boolean, optional
            True to retrieve the color boundary image for a Glioblastoma SectionImage.
        Returns
        -------
        url : string
            The constructed URL
            
        Notes
        -----
        'downsample=1' halves the number of pixels of the original image
        both horizontally and vertically.
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
        '''
        params = []
        
        downsample = kwargs.get('downsample', None)
        
        if downsample != None:
            params.append('downsample=%d' % (downsample))
        
        quality = kwargs.get('quality', None)
        
        if quality != None:
            params.append('quality=%d' % (quality))
        
        expression = kwargs.get('expression', None)
        
        if expression != None:
            if expression:
                params.append('expression=true')
            else:
                params.append('expression=false')
        
        # region of interest
        for roi_key in [ 'left', 'top', 'width', 'height']:
            roi_value = kwargs.get(roi_key, None)
            if roi_value != None:
                params.append('%s=%d' % (roi_key, roi_value))
        
        range_list = kwargs.get('range', None)
        
        # see http://api.brain-map.org/api/v2/data/SectionDataSet/100141599.xml?include=equalization,section_images
        if range_list:
            params.append('range=%s' % (','.join(str(r) for r in range_list)))
        
        tumor_feature_annotation = kwargs.get('tumor_feature_annotation', None)
        
        if tumor_feature_annotation != None:
            if tumor_feature_annotation:
                params.append('tumor_feature_annotation=true')
            else:
                params.append('tumor_feature_annotation=false')
        
        tumor_feature_boundary = kwargs.get('tumor_feature_boundary', None)
        
        if tumor_feature_boundary != None:
            if tumor_feature_boundary:
                params.append('tumor_feature_boundary=true')
            else:
                params.append('tumor_feature_boundary=false')
        
        if len(params) > 0:
            url_params = "?" + "&".join(params)
        else:
            url_params = ''
        
        url = ''.join([self.section_image_download_endpoint,
                       '/',
                       str(section_image_id),
                       url_params])
        
        return url
    
    
    def build_atlas_image_url(self,
                              atlas_image_id,
                              **kwargs):
        # see http://help.brain-map.org/display/api/Downloading+an+Image#DownloadinganImage-ProjectionImage%26nbsp%3BDownloadService
        raise(Exception('unimplemented'))
    
    
    def build_projection_image_url(self,
                                   projection_image_id,
                                   **kwargs):
        # see http://help.brain-map.org/display/api/Downloading+an+Image#DownloadinganImage-ProjectionImage%26nbsp%3BDownloadService
        raise(Exception('unimplemented'))
    
    
    def read_data(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def download_section_image(self,
                               section_image_id,
                               file_path=None,
                               **kwargs):
        '''
        Parameters
        ----------
        section_image_id : integer
            What to download.
        file_path : string, optional
            Where to put it.  <section_image_id>.jpg (default).
        '''
        if file_path == None:
            file_path = '%d.jpg' % (section_image_id)
        
        image_url = self.build_section_image_url(section_image_id,
                                                 **kwargs)
        print(image_url)
        self.retrieve_file_over_http(image_url, file_path)
    
    
    def download_atlas_image(self,
                             atlas_image_id,
                             file_path=None,
                             **kwargs):
        if file_path == None:
            file_path = '%d.jpg' % (atlas_image_id)
        
        image_url = self.build_atlas_image_url(atlas_image_id,
                                               **kwargs)
        print(image_url)
        self.retrieve_file_over_http(image_url, file_path)
    
    
    def download_projection_image(self,
                                  projection_image_id,
                                  file_path=None,
                                  **kwargs):
        if file_path == None:
            file_path = '%d.jpg' % (projection_image_id)
        
        image_url = self.build_projection_image_url(projection_image_id,
                                                    **kwargs)
        print(image_url)
        self.retrieve_file_over_http(image_url, file_path)



if __name__ == '__main__':
    # queries from http://help.brain-map.org/display/api/Downloading+an+Image#DownloadinganImage-ProjectionImage%26nbsp%3BDownloadService
    import json
    from allensdk.api.queries.image.image_download_api import ImageDownloadApi
    
    a = ImageDownloadApi()
    #a.download_section_image(69750516, downsample=4)
    #a.download_section_image(69750516, downsample=4, expression=True)
    #a.download_section_image(69750516, left=6174, top=2282, width=1000, height=1000)
    #a.download_section_image(69750516, downsample=3, quality=50)
    # TODO: http://api.brain-map.org/api/v2/data/SectionDataSet/100141599.xml?include=equalization,section_images
    #a.download_section_image(102146167, range=[0, 923, 0, 987, 0, 4095], downsample=4)
    #a.download_section_image(311175878, downsample=4, tumor_feature_annotation=True)
    #a.download_section_image(311174547, downsample=4, tumor_feature_boundary=True)
    #a.download_section_image(71592412)
    
    

