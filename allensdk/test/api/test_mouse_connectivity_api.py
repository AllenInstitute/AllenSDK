# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2016-2017. Allen Institute. All rights reserved.
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
import os
import pytest
from mock import patch, Mock
import itertools as it
import numpy as np
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi as MCA

MOCK_ANNOTATION_DATA = 'mock_annotation_data'
MOCK_ANNOTATION_IMAGE = 'mock_annotation_image'
DOWNLOAD_LINK = '/path/to/link'


@pytest.fixture
def connectivity():
    mca = MCA()
    
    return mca


def CCF_VERSIONS():
    return [MCA.CCF_2015,
            MCA.CCF_2016]


def DATA_PATHS(): 
    return [MCA.AVERAGE_TEMPLATE,
            MCA.ARA_NISSL,
            MCA.MOUSE_2011,
            MCA.DEVMOUSE_2012,
            MCA.CCF_2015,
            MCA.CCF_2016]


def RESOLUTIONS():
    return [MCA.VOXEL_RESOLUTION_10_MICRONS,
            MCA.VOXEL_RESOLUTION_25_MICRONS,
            MCA.VOXEL_RESOLUTION_50_MICRONS,
            MCA.VOXEL_RESOLUTION_100_MICRONS]


@pytest.mark.parametrize("data_path,resolution",
                         it.product(DATA_PATHS(),
                                    RESOLUTIONS()))
@patch.object(MCA, "retrieve_file_over_http")
def test_download_volumetric_data(mock_retrieve,
                                  connectivity,
                                  data_path,
                                  resolution):
    cache_filename = "annotation_%d.nrrd" % (resolution)

    connectivity.download_volumetric_data(data_path,
                                          cache_filename,
                                          resolution)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (data_path,
         resolution),
        cache_filename)


@pytest.mark.parametrize("ccf_version,resolution",
                         it.product(CCF_VERSIONS(),
                                    RESOLUTIONS()))
@patch.object(MCA, "retrieve_file_over_http")
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_annotation_volume(os_makedirs,
                                    nrrd_read,
                                    mock_retrieve,
                                    connectivity,
                                    ccf_version,
                                    resolution):
    cache_file = '/path/to/annotation_%d.nrrd' % (resolution)

    connectivity.download_annotation_volume(
        ccf_version,
        resolution,
        cache_file,
        reader=nrrd_read)

    nrrd_read.assert_called_once_with(cache_file)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (ccf_version,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch.object(MCA, "retrieve_file_over_http")
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_annotation_volume_default(os_makedirs,
                                            nrrd_read,
                                            mock_retrieve,
                                            connectivity,
                                            resolution):
    a, b = connectivity.download_annotation_volume(
        None,
        resolution,
        '/path/to/annotation_%d.nrrd' % (resolution),
        reader=nrrd_read)
    
    assert a
    assert b

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (MCA.CCF_VERSION_DEFAULT,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch.object(MCA, "retrieve_file_over_http")
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_structure_mask(os_makedirs, 
                                 nrrd_read, 
                                 mock_retrieve,
                                 connectivity, 
                                 resolution):

    structure_id = 12

    a, b = connectivity.download_structure_mask(structure_id,
                                                None,
                                                resolution,'/path/to/foo.nrrd',
                                                reader=nrrd_read)

    assert a
    assert b

    expected = 'http://download.alleninstitute.org/informatics-archive/'\
               'current-release/mouse_ccf/{0}/structure_masks/'\
               'structure_masks_{1}/structure_{2}.nrrd'.format(MCA.CCF_VERSION_DEFAULT, 
                                                               resolution, 
                                                               structure_id)
    mock_retrieve.assert_called_once_with(expected, '/path/to/foo.nrrd')
    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch.object(MCA, "retrieve_file_over_http")
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_template_volume(os_makedirs,
                                  nrrd_read,
                                  mock_retrieve,
                                  connectivity,
                                  resolution):
    connectivity.download_template_volume(
        resolution,
        '/path/to/average_template_%d.nrrd' % (resolution),
        reader=nrrd_read)

    nrrd_read.assert_called_once_with('/path/to/average_template_%d.nrrd' % (resolution))

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/average_template/average_template_%d.nrrd" % 
        (resolution),
        "/path/to/average_template_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@patch.object(MCA, "json_msg_query")
def test_get_experiments_no_ids(mock_query,
                                connectivity):
    connectivity.get_experiments(None)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31]")


@patch.object(MCA, "json_msg_query")
def test_get_experiments_one_id(mock_query,
                                connectivity):
    connectivity.get_experiments(987)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31],[id$in987]")


@patch.object(MCA, "json_msg_query")
def test_get_experiments_ids(mock_query,
                             connectivity):
    connectivity.get_experiments([9,8,7])

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31],[id$in9,8,7]")


@patch.object(MCA, "json_msg_query")
def test_get_manual_injection_summary(mock_query,
                                      connectivity):
    connectivity.get_manual_injection_summary(123)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[id$in123],"
        "rma::include,specimen(donor(transgenic_mouse(transgenic_lines)),"
        "injections(structure,age)),equalization,products,"
        "rma::options[only$eqid,failed,storage_directory,red_lower,red_upper,"
        "green_lower,green_upper,blue_lower,blue_upper,products.id,"
        "specimen_id,structure_id,reference_space_id,"
        "primary_injection_structure_id,registration_point,coordinates_ap,"
        "coordinates_dv,coordinates_ml,angle,sex,strain,injection_materials,"
        "acronym,structures.name,days,transgenic_mice.name,"
        "transgenic_lines.name,transgenic_lines.description,"
        "transgenic_lines.id,donors.id]")


@patch.object(MCA, "json_msg_query")
def test_get_experiment_detail(mock_query,
                               connectivity):
    connectivity.get_experiment_detail(123)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[id$eq123],"
        "rma::include,specimen(stereotaxic_injections"
        "(primary_injection_structure,structures,"
        "stereotaxic_injection_coordinates)),equalization,sub_images,"
        "rma::options[order$eq'sub_images.section_number$asc']")


@patch.object(MCA, "json_msg_query")
def test_get_projection_image_info(mock_query,
                                   connectivity):
    connectivity.get_projection_image_info(123, 456)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[id$eq123],rma::include,"
        "equalization,sub_images[section_number$eq456]")


def test_build_reference_aligned_channel_volumes_url(connectivity):
    url = \
        connectivity.build_reference_aligned_image_channel_volumes_url(123456)

    assert url == ("http://api.brain-map.org/api/v2/data/query.json?q="
                   "model::WellKnownFile,rma::criteria,"
                   "well_known_file_type[name$eq'ImagesResampledTo25MicronARA']"
                   "[attachable_id$eq123456]")


@patch.object(MCA, "retrieve_file_over_http")
@patch.object(MCA, "do_query", return_value=DOWNLOAD_LINK)
def test_reference_aligned_channel_volumes(mock_query,
                                           mock_retrieve,
                                           connectivity):
    connectivity.download_reference_aligned_image_channel_volumes(123456)

    mock_retrieve.assert_called_once_with(
        "http://api.brain-map.org/path/to/link",
        "123456.zip")


@patch.object(MCA, "json_msg_query")
def test_experiment_source_search(mock_query,
                                  connectivity):
    connectivity.experiment_source_search(
        injection_structures='Isocortex',
        primary_structure_only=True)

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_injection_structure"
        "[injection_structures$eqIsocortex][primary_structure_only$eqtrue]")


@patch.object(MCA, "json_msg_query")
def test_experiment_spatial_search(mock_query,
                                   connectivity):
    connectivity.experiment_spatial_search(
        seed_point=[6900,5050,6450])

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_target_spatial"
        "[seed_point$eq6900,5050,6450]")


@patch.object(MCA, "json_msg_query")
def test_injection_coordinate_search(mock_query,
                                     connectivity):
    connectivity.experiment_injection_coordinate_search(
        seed_point=[6900,5050,6450])

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_injection_coordinate"
        "[seed_point$eq6900,5050,6450]")


@patch.object(MCA, "json_msg_query")
def test_experiment_correlation_search(mock_query,
                                       connectivity):
    connectivity.experiment_correlation_search(
        row=112670853, structure='TH')

    mock_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_correlation"
        "[row$eq112670853][structure$eqTH]")


@pytest.mark.parametrize("injection,hemisphere",
                         it.product([True, False,None],
                                    [['left'],['right'],None]))
@patch.object(MCA, "json_msg_query")
def test_get_structure_unionizes(mock_query,
                                 connectivity,
                                 injection,
                                 hemisphere):
    connectivity.get_structure_unionizes(
        experiment_ids=[126862385],
        is_injection=injection,
        hemisphere_ids=hemisphere,
        include='structure')

    i = ''

    if injection is not None:
        i = "[is_injection$eq%s]" % (str(injection).lower())

    h = ''

    if hemisphere is not None:
        h = "[hemisphere_id$in%s]" % (hemisphere[0])

    mock_query.assert_called_once_with(
        ("http://api.brain-map.org/api/v2/data/query.json?q="
         "model::ProjectionStructureUnionize,rma::criteria,"
         "[section_data_set_id$in126862385]%s%s,"
         "rma::include,structure,rma::options[num_rows$eq'all']"
         "[count$eqfalse]") % (i, h))


def test_download_injection_density(connectivity):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        connectivity.download_injection_density(
            'file.name', 12345, 10)

        gda.assert_called_once_with(
            "http://api.brain-map.org/grid_data/download_file/"
            "12345"
            "?image=injection_density&resolution=10",
            "file.name")


def test_download_projection_density(connectivity):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        connectivity.download_projection_density(
            'file.name', 12345, 10)

        gda.assert_called_once_with(
            "http://api.brain-map.org/grid_data/download_file/"
            "12345"
            "?image=projection_density&resolution=10",
            "file.name")


def test_download_data_mask_density(connectivity):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        connectivity.download_data_mask(
            'file.name', 12345, 10)

        gda.assert_called_once_with(
            "http://api.brain-map.org/grid_data/download_file/"
            "12345"
            "?image=data_mask&resolution=10",
            "file.name")


def test_download_injection_fraction(connectivity):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        connectivity.download_injection_fraction(
            'file.name', 12345, 10)

        gda.assert_called_once_with(
            "http://api.brain-map.org/grid_data/download_file/"
            "12345"
            "?image=injection_fraction&resolution=10",
            "file.name")


def test_calculate_injection_centroid(connectivity):
    density = np.array(([1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0]))
    fraction = np.array(([1.0,1.0,1.0,1.0],
                         [1.0,1.0,1.0,1.0],
                         [1.0,1.0,1.0,1.0],
                         [1.0,1.0,1.0,1.0]))

    centroid = connectivity.calculate_injection_centroid(
        density, fraction, resolution=25)
    
    assert np.array_equal(centroid, [37.5, 37.5])
