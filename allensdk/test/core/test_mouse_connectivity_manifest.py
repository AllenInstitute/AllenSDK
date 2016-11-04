import pytest
from mock import Mock, MagicMock, patch
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import os


@pytest.fixture
def mcc():
    mcc = MouseConnectivityCache(
        resolution=MouseConnectivityApi.VOXEL_RESOLUTION_100_MICRONS,
        manifest_file='mcc_manifest.json')
    mcc.api.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')

    return mcc


@pytest.fixture
def unmocked_mcc():
    mcc = MouseConnectivityCache(
        resolution=MouseConnectivityApi.VOXEL_RESOLUTION_100_MICRONS,
        ccf_version=MouseConnectivityApi.CCF_2015)

    return mcc


@pytest.fixture
def mcc_old():
    mcc_old = MouseConnectivityCache(
        resolution=MouseConnectivityApi.VOXEL_RESOLUTION_100_MICRONS,
        ccf_version=MouseConnectivityApi.CCF_2015,
        manifest_file='mcc_manifest.json')
    mcc_old.api.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')

    return mcc_old


def test_get_annotation_volume_2015(mcc_old):
    with patch('os.path.exists', Mock(return_value=False)):
        with patch('allensdk.config.manifest.Manifest.safe_mkdir'):
            with patch('os.makedirs'):
                with patch('nrrd.read',
                           Mock(return_value=('a', 'b'))):
                    mcc_old.get_annotation_volume(file_name="/tmp/n100.nrrd")
    
                    mcc_old.api.retrieve_file_over_http.assert_called_once_with(
                        'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2015/annotation_100.nrrd',
                        '/tmp/n100.nrrd')


def test_get_annotation_volume(mcc):
    with patch('os.path.exists', Mock(return_value=False)):
        with patch('allensdk.config.manifest.Manifest.safe_mkdir'):
            with patch('os.makedirs'):
                with patch('nrrd.read',
                           Mock(return_value=('a', 'b'))):
                    mcc.get_annotation_volume(file_name="/tmp/n100.nrrd")
    
                    mcc.api.retrieve_file_over_http.assert_called_once_with(
                        'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2016/annotation_100.nrrd',
                        '/tmp/n100.nrrd')


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_notebook(unmocked_mcc):
    mcc = unmocked_mcc
    all_experiments = mcc.get_experiments(dataframe=True)
    # all_experiments.loc[122642490]

    ontology = mcc.get_ontology()
    isocortex = ontology['Isocortex']

    cre_cortical_experiments = mcc.get_experiments(
        cre=True, injection_structure_ids=isocortex['id'])

    rbp4_cortical_experiments = mcc.get_experiments(
        cre=[ 'Rbp4-Cre_KL100' ], injection_structure_ids=isocortex['id'])

    visp = ontology['VISp']
    visp_experiments = mcc.get_experiments(cre=False, 
                                           injection_structure_ids=visp['id'])

    structure_unionizes = mcc.get_structure_unionizes(
        [ e['id'] for e in visp_experiments ], 
        is_injection=False,
        structure_ids=isocortex['id'].tolist())

    dense_unionizes = structure_unionizes[ structure_unionizes.projection_density > .5 ]
    large_unionizes = dense_unionizes[ dense_unionizes.volume > .5 ]
    large_structures = ontology[large_unionizes.structure_id]

    visp_experiment_ids = [ e['id'] for e in visp_experiments ]
    ctx_children = ontology.get_child_ids( ontology['Isocortex'].id )
    
    pm = mcc.get_projection_matrix(experiment_ids=visp_experiment_ids, 
                                   projection_structure_ids=ctx_children,
                                   hemisphere_ids=[2],  # right hemisphere, ipsilateral
                                   parameter='projection_density')

    experiment_id = 181599674

    # projection density: number of projecting pixels / voxel volume
    pd, pd_info = mcc.get_projection_density(experiment_id)
    
    # injection density: number of projecting pixels in injection site / voxel volume
    ind, ind_info = mcc.get_injection_density(experiment_id)
    
    # injection fraction: number of pixels in injection site / voxel volume
    inf, inf_info = mcc.get_injection_fraction(experiment_id)
    
    # data mask:
    # binary mask indicating which voxels contain valid data
    dm, dm_info = mcc.get_data_mask(experiment_id)
    
    template, template_info = mcc.get_template_volume()
    annot, annot_info = mcc.get_annotation_volume()

    pd_mip = pd.max(axis=0)
    ind_mip = ind.max(axis=0)
    inf_mip = inf.max(axis=0)
    dm_mip = dm.min(axis=0)

    isocortex_mask, _ = mcc.get_structure_mask(isocortex['id'])

    # pull out the values of all voxels in the isocortex mask
    isocortex_pd = pd[isocortex_mask > 0]
    
    # print out the average projection density of voxels in the isocortex
    assert np.isclose(isocortex_pd.mean(), 0.0194823)
