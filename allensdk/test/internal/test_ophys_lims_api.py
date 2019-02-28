import pytest

from allensdk.internal.api import OneResultExpectedError, OneOrMoreResultExpectedError
from allensdk.internal.api.ophys_lims_api import OphysLimsApi

@pytest.fixture(scope="function")
def ophys_lims_api():
    return OphysLimsApi()


@pytest.fixture(scope="function")
def api_data():
     return {702134928:
                      {'ophys_dir':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/',
                       'demix_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix/702134928_demixed_traces.h5',
                       'avgint_a1X_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/avgInt_a1X.png',
                       'rigid_motion_transform_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/702134928_rigid_motion_transform.csv',
                       'targeted_structure':'VISal',
                       'imaging_depth':175,
                       'stimulus_name':None,
                       'reporter_line':'Ai148(TIT2L-GC6f-ICL-tTA2)',
                       'driver_line':['Vip-IRES-Cre'],
                       'LabTracks_ID':'363887',
                       'full_genotype':'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt'
                       }
            }


def expected_fail(func, *args, **kwargs):
    expected_fail = False
    try:
        func(*args, **kwargs)
    except (OneResultExpectedError, OneOrMoreResultExpectedError) as e:
        expected_fail = True
    
    assert expected_fail == True


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_ophys_experiment_dir(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_ophys_experiment_dir
    key = 'ophys_dir'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)
        


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_demix_file(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_demix_file
    key = 'demix_file'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_avgint_a1X_file(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_avgint_a1X_file
    key = 'avgint_a1X_file'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_rigid_motion_transform_file(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_rigid_motion_transform_file
    key = 'rigid_motion_transform_file'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_targeted_structure(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_targeted_structure
    key = 'targeted_structure'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_imaging_depth(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_imaging_depth
    key = 'imaging_depth'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_stimulus_name(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_stimulus_name
    key = 'stimulus_name'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_reporter_line(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_reporter_line
    key = 'reporter_line'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_driver_line(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_driver_line
    key = 'driver_line'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_LabTracks_ID(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_LabTracks_ID
    key = 'LabTracks_ID'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id', [702134928, 0])
def test_get_full_genotype(ophys_experiment_id, api_data, ophys_lims_api):
    f = ophys_lims_api.get_full_genotype
    key = 'full_genotype'
    if ophys_experiment_id in api_data:
        assert f(ophys_experiment_id=ophys_experiment_id) == api_data[ophys_experiment_id][key]
    else:
        expected_fail(f, ophys_experiment_id=ophys_experiment_id)










@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id, compare_val', [
    pytest.param(511458874, '/allen/programs/braintv/production/neuralcoding/prod6/specimen_503292442/ophys_experiment_511458874/511458874.nwb'),
    pytest.param(0, None)
])
def test_get_nwb_filepath(ophys_experiment_id, compare_val, ophys_lims_api):

    if compare_val is None:
        expected_fail = False
        try:
            ophys_lims_api.get_nwb_filepath(ophys_experiment_id)
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail == True
    else:
        assert ophys_lims_api.get_nwb_filepath(ophys_experiment_id=ophys_experiment_id) == compare_val


        # 'demix_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix/702134928_demixed_traces.h5',
        # 'avgint_a1X_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/avgInt_a1X.png',
        # 'rigid_motion_transform_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/702134928_rigid_motion_transform.csv',
        # 'targeted_structure':'VISal',
        # 'imaging_depth':175,
        # 'stimulus_name':None,
        # 'reporter_line':'Ai148(TIT2L-GC6f-ICL-tTA2)',
        # 'driver_line':['Vip-IRES-Cre'],
        # 'LabTracks_ID':'363887',
        # 'full_genotype':'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt'
        # }