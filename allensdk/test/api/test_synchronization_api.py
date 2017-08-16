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
