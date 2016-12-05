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

from ..api import Api


class SynchronizationApi(Api):
    '''HTTP client for image synchronization services uses the image alignment results from
    the Informatics Data Processing Pipeline.
    Note: all locations on SectionImages are reported in pixel coordinates
    and all locations in 3-D ReferenceSpaces are reported in microns.

    See `Image to Image Synchronization <http://help.brain-map.org/display/api/Image-to-Image+Synchronization>`_
    for additional documentation.
    '''

    def __init__(self, base_uri=None):
        super(SynchronizationApi, self).__init__(base_uri)

    def get_image_to_atlas(self,
                           section_image_id,
                           x, y,
                           atlas_id):
        '''For a specified Atlas, find the closest annotated SectionImage
        and (x,y) location as defined by a seed SectionImage and seed (x,y) location.

        Parameters
        ----------
        section_image_id : integer
            Seed for spatial sync.
        x : float
            Pixel coordinate of the seed location in the seed SectionImage.
        y : float
            Pixel coordinate of the seed location in the seed SectionImage.
        atlas_id : int
            Target Atlas for image sync.

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.image_to_atlas_endpoint,
                       '/',
                       str(section_image_id),
                       '.json',
                       '?x=%f&y=%f' % (x, y),
                       '&atlas_id=',
                       str(atlas_id)])

        return self.json_msg_query(url)

    def get_image_to_image(self,
                           section_image_id,
                           x, y,
                           section_data_set_ids):
        '''For a list of target SectionDataSets, find the closest SectionImage
        and (x,y) location as defined by a seed SectionImage and seed (x,y) pixel location.

        Parameters
        ----------
        section_image_id : integer
            Seed for spatial sync.
        x : float
            Pixel coordinate of the seed location in the seed SectionImage.
        y : float
            Pixel coordinate of the seed location in the seed SectionImage.
        section_data_set_ids : list of integers
            Target SectionDataSet IDs for image sync.

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.image_to_image_endpoint,
                       '/',
                       str(section_image_id),
                       '.json',
                       '?x=%f&y=%f' % (x, y),
                       '&section_data_set_ids=',
                       ','.join(str(i) for i in section_data_set_ids)])

        return self.json_msg_query(url)

    def get_image_to_image_2d(self,
                              section_image_id,
                              x, y,
                              section_image_ids):
        '''For a list of target SectionImages, find the closest (x,y) location
        as defined by a seed SectionImage and seed (x,y) location.

        Parameters
        ----------
        section_image_id : integer
            Seed for image sync.
        x : float
            Pixel coordinate of the seed location in the seed SectionImage.
        y : float
            Pixel coordinate of the seed location in the seed SectionImage.
        section_image_ids : list of ints
            Target SectionImage IDs for image sync.

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.image_to_image_2d_endpoint,
                       '/',
                       str(section_image_id),
                       '.json',
                       '?x=%f&y=%f' % (x, y),
                       '&section_image_ids=',
                       ','.join(str(i) for i in section_image_ids)])

        return self.json_msg_query(url)

    def get_reference_to_image(self,
                               reference_space_id,
                               x, y, z,
                               section_data_set_ids):
        '''For a list of target SectionDataSets, find the closest SectionImage
        and (x,y) location as defined by a (x,y,z) location in a specified ReferenceSpace.

        Parameters
        ----------
        reference_space_id : integer
            Seed for spatial sync.
        x : float
            Coordinate (in microns) of the seed location in the seed ReferenceSpace.
        y : float
            Coordinate (in microns) of the seed location in the seed ReferenceSpace.
        z : float
            Coordinate (in microns) of the seed location in the seed ReferenceSpace.
        section_data_set_ids : list of ints
            Target SectionDataSets IDs for image sync.

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.reference_to_image_endpoint,
                       '/',
                       str(reference_space_id),
                       '.json',
                       '?x=%f&y=%f&z=%f' % (x, y, z),
                       '&section_data_set_ids=',
                       ','.join(str(i) for i in section_data_set_ids)])

        return self.json_msg_query(url)

    def get_image_to_reference(self,
                               section_image_id,
                               x, y):
        '''For a specified SectionImage and (x,y) location,
        return the (x,y,z) location in the ReferenceSpace of the associated SectionDataSet.

        Parameters
        ----------
        section_image_id : integer
            Seed for image sync.
        x : float
            Pixel coordinate on the specified SectionImage.
        y : float
            Pixel coordinate on the specified SectionImage.

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.image_to_reference_endpoint,
                       '/',
                       str(section_image_id),
                       '.json',
                       '?x=%f&y=%f' % (x, y)])

        return self.json_msg_query(url)

    def get_structure_to_image(self,
                               section_data_set_id,
                               structure_ids):
        '''For a list of target structures, find the closest SectionImage
        and (x,y) location as defined by the centroid of each Structure.

        Parameters
        ----------
        section_data_set_id : integer
            primary key
        structure_ids : list of integers
            primary key

        Returns
        -------
        dict
            The parsed json response
        '''
        url = ''.join([self.structure_to_image_endpoint,
                       '/',
                       str(section_data_set_id),
                       '.json',
                       '?structure_ids=',
                       ','.join([str(i) for i in structure_ids])])

        return self.json_msg_query(url)
