# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import pytest
from mock import MagicMock
from allensdk.api.queries.synchronization_api import SynchronizationApi


@pytest.fixture
def synch():
    sa = SynchronizationApi()
    sa.json_msg_query = MagicMock(name='json_msg_query')

    return sa


def test_image_to_image(synch):
    '''
    Notes
    -----
    Expected link is slightly modified for json and float serialization of zeros.

    See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
    , link labeled 'Sync a VISp and VISal experiment to a location in a SCs SectionDataSet'.
    '''
    section_image_id = 114754496
    (x, y) = (18232, 10704)
    section_data_set_ids = [113887162, 116903968]

    _ = synch.get_image_to_image(section_image_id,
                                 x, y,
                                 section_data_set_ids)
    expected = 'http://api.brain-map.org/api/v2/image_to_image/114754496.json?x=18232.000000&y=10704.000000&section_data_set_ids=113887162,116903968'
    synch.json_msg_query.assert_called_once_with(expected)


def test_image_to_image_2d(synch):
    section_image_id = 68173101
    (x, y) = (6208, 2368)
    section_image_ids = [68173103, 68173105, 68173107]

    _ = synch.get_image_to_image_2d(section_image_id,
                                    x, y,
                                    section_image_ids)
    expected = 'http://api.brain-map.org/api/v2/image_to_image_2d/68173101.json?x=6208.000000&y=2368.000000&section_image_ids=68173103,68173105,68173107'
    synch.json_msg_query.assert_called_once_with(expected)


def test_reference_to_image(synch):
    reference_space_id = 10
    (x, y, z) = (6085, 3670, 4883)
    section_data_set_ids = [68545324, 67810540]

    _ = synch.get_reference_to_image(reference_space_id,
                                     x, y, z,
                                     section_data_set_ids)
    expected = 'http://api.brain-map.org/api/v2/reference_to_image/10.json?x=6085.000000&y=3670.000000&z=4883.000000&section_data_set_ids=68545324,67810540'
    synch.json_msg_query.assert_called_once_with(expected)


def test_image_to_reference(synch):
    section_image_id = 68173101
    (x, y) = (6208, 2368)

    _ = synch.get_image_to_reference(section_image_id,
                                     x, y)
    expected = 'http://api.brain-map.org/api/v2/image_to_reference/68173101.json?x=6208.000000&y=2368.000000'
    synch.json_msg_query.assert_called_once_with(expected)


def test_structure_to_image(synch):
    section_data_set_id = 68545324
    structure_ids = [315, 698, 1089, 703, 477,
                     803, 512, 549, 1097, 313, 771, 354]

    _ = synch.get_structure_to_image(section_data_set_id,
                                     structure_ids)
    expected = 'http://api.brain-map.org/api/v2/structure_to_image/68545324.json?structure_ids=315,698,1089,703,477,803,512,549,1097,313,771,354'
    synch.json_msg_query.assert_called_once_with(expected)


def test_image_to_atlas(synch):
    '''
    Notes
    -----
    Expected link is slightly modified for json and float serialization of zeros.

    See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
    , link labeled 'Sync the P56 coronal reference atlas to a location in the SCs SectionDataSet'.
    '''
    section_image_id = 114754496
    (x, y) = (18232, 10704)
    atlas_id = 1
    _ = synch.get_image_to_atlas(section_image_id,
                                 x, y,
                                 atlas_id)
    expected = 'http://api.brain-map.org/api/v2/image_to_atlas/114754496.json?x=18232.000000&y=10704.000000&atlas_id=1'
    synch.json_msg_query.assert_called_once_with(expected)
