# Copyright 2016-2017 Allen Institute for Brain Science
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
import pytest
from mock import patch, Mock
import itertools as it
import numpy as np


try:
    reload
except NameError:
    try:
        from importlib import reload
    except ImportError:
        from imp import reload

        
@pytest.fixture
def nrrd_read():
    with patch('nrrd.read',
               Mock(name='nrrd_read_file_mcm',
                    return_value=('mock_annotation_data',
                                  'mock_annotation_image'))) as nrrd_read:
        import allensdk.core.mouse_connectivity_cache as MCC
        import allensdk.api.queries.mouse_connectivity_api as MCA
        reload(MCC)
        reload(MCA)

    return nrrd_read

@pytest.fixture
def MCA(nrrd_read):
    import allensdk.api.queries.mouse_connectivity_api as MCA

    download_link = '/path/to/link'
    MCA.MouseConnectivityApi.do_query = Mock(return_value=download_link)
    MCA.MouseConnectivityApi.json_msg_query = Mock(name='json_msg_query')
    MCA.MouseConnectivityApi.retrieve_file_over_http = \
        Mock(name='retrieve_file_over_http')
    
    return MCA

@pytest.fixture
def connectivity(MCA):
    mca = MCA.MouseConnectivityApi()
    
    return mca


@pytest.fixture
def MCC(nrrd_read):
    import allensdk.core.mouse_connectivity_cache as MCC

    return MCC


def CCF_VERSIONS():
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
     
    return [MouseConnectivityApi.CCF_2015,
            MouseConnectivityApi.CCF_2016]


def DATA_PATHS(): 
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

    return [MouseConnectivityApi.AVERAGE_TEMPLATE,
            MouseConnectivityApi.ARA_NISSL,
            MouseConnectivityApi.MOUSE_2011,
            MouseConnectivityApi.DEVMOUSE_2012,
            MouseConnectivityApi.CCF_2015,
            MouseConnectivityApi.CCF_2016]


def RESOLUTIONS():
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

    return [MouseConnectivityApi.VOXEL_RESOLUTION_10_MICRONS,
            MouseConnectivityApi.VOXEL_RESOLUTION_25_MICRONS,
            MouseConnectivityApi.VOXEL_RESOLUTION_50_MICRONS,
            MouseConnectivityApi.VOXEL_RESOLUTION_100_MICRONS]

MOCK_ANNOTATION_DATA = 'mock_annotation_data'
MOCK_ANNOTATION_IMAGE = 'mock_annotation_image'


@pytest.mark.parametrize("data_path,resolution",
                         it.product(DATA_PATHS(),
                                    RESOLUTIONS()))
def test_download_volumetric_data(nrrd_read,
                                  connectivity,
                                  data_path,
                                  resolution):
    cache_filename = "annotation_%d.nrrd" % (resolution)

    nrrd_read.reset_mock()
    connectivity.retrieve_file_over_http.reset_mock()

    connectivity.download_volumetric_data(data_path,
                                          cache_filename,
                                          resolution)

    connectivity.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (data_path,
         resolution),
        cache_filename)


@pytest.mark.parametrize("ccf_version,resolution",
                         it.product(CCF_VERSIONS(),
                                    RESOLUTIONS()))
@patch('os.makedirs')
def test_download_annotation_volume(os_makedirs,
                                    nrrd_read,
                                    connectivity,
                                    ccf_version,
                                    resolution):
    nrrd_read.reset_mock()
    connectivity.retrieve_file_over_http.reset_mock()

    cache_file = '/path/to/annotation_%d.nrrd' % (resolution)

    connectivity.download_annotation_volume(
        ccf_version,
        resolution,
        cache_file)

    nrrd_read.assert_called_once_with(cache_file)

    connectivity.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (ccf_version,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch('os.makedirs')
def test_download_annotation_volume_default(os_makedirs,
                                            nrrd_read,
                                            MCA,
                                            connectivity,
                                            resolution):
    connectivity.retrieve_file_over_http.reset_mock()

    a, b = connectivity.download_annotation_volume(
        None,
        resolution,
        '/path/to/annotation_%d.nrrd' % (resolution),
        reader=nrrd_read)
    
    assert a
    assert b

    print(connectivity.retrieve_file_over_http.call_args_list)

    connectivity.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (MCA.MouseConnectivityApi.CCF_VERSION_DEFAULT,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch('os.makedirs')
def test_download_template_volume(os_makedirs,
                                  connectivity,
                                  resolution):
    connectivity.retrieve_file_over_http.reset_mock()

    connectivity.download_template_volume(
        resolution,
        '/path/to/average_template_%d.nrrd' % (resolution))

    connectivity.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/average_template/average_template_%d.nrrd" % 
        (resolution),
        "/path/to/average_template_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


def test_get_experiments_no_ids(connectivity):
    connectivity.get_experiments(None)

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31]")


def test_get_experiments_one_id(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.get_experiments(987)

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31],[id$in987]")


def test_get_experiments_ids(connectivity):
    connectivity.json_msg_query.reset_mock()
    
    connectivity.get_experiments([9,8,7])

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[failed$eqfalse],"
        "products[id$in5,31],[id$in9,8,7]")


def test_get_manual_injection_summary(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.get_manual_injection_summary(123)

    connectivity.json_msg_query.assert_called_once_with(
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


def test_get_experiment_detail(connectivity):
    connectivity.json_msg_query.reset_mock()
    
    connectivity.get_experiment_detail(123)

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::SectionDataSet,rma::criteria,[id$eq123],"
        "rma::include,specimen(stereotaxic_injections"
        "(primary_injection_structure,structures,"
        "stereotaxic_injection_coordinates)),equalization,sub_images,"
        "rma::options[order$eq'sub_images.section_number$asc']")


def test_get_projection_image_info(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.get_projection_image_info(123, 456)

    connectivity.json_msg_query.assert_called_once_with(
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


def test_reference_aligned_channel_volumes(connectivity):
    connectivity.retrieve_file_over_http.reset_mock()

    connectivity.download_reference_aligned_image_channel_volumes(123456)

    connectivity.retrieve_file_over_http.assert_called_once_with(
        "http://api.brain-map.org/path/to/link",
        "123456.zip")


def test_experiment_source_search(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.experiment_source_search(
        injection_structures='Isocortex',
        primary_structure_only=True)

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_injection_structure"
        "[injection_structures$eqIsocortex][primary_structure_only$eqtrue]")


def test_experiment_spatial_search(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.experiment_spatial_search(
        seed_point=[6900,5050,6450])

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_target_spatial"
        "[seed_point$eq6900,5050,6450]")


def test_injection_coordinate_search(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.experiment_injection_coordinate_search(
        seed_point=[6900,5050,6450])

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_injection_coordinate"
        "[seed_point$eq6900,5050,6450]")


def test_experiment_correlation_search(connectivity):
    connectivity.json_msg_query.reset_mock()

    connectivity.experiment_correlation_search(
        row=112670853, structure='TH')

    connectivity.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "service::mouse_connectivity_correlation"
        "[row$eq112670853][structure$eqTH]")


@pytest.mark.parametrize("injection,hemisphere",
                         it.product([True, False,None],
                                    [['left'],['right'],None]))
def test_get_structure_unionizes(connectivity,
                                 injection,
                                 hemisphere):
    connectivity.json_msg_query.reset_mock()

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

    connectivity.json_msg_query.assert_called_once_with(
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


@pytest.mark.run('last')
def test_cleanup():
    import allensdk.api.queries.mouse_connectivity_api
    reload(allensdk.api.queries.mouse_connectivity_api)
    import allensdk.core.mouse_connectivity_cache
    reload(allensdk.core.mouse_connectivity_cache)
