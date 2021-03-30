import math

import numpy as np
import pandas as pd
import pynwb
import pytest

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorNwbApi)
from allensdk.brain_observatory.behavior.stimulus_processing import \
    StimulusTemplate, get_stimulus_templates

from allensdk.brain_observatory.behavior.write_behavior_nwb.__main__ import write_behavior_nwb  # noqa: E501


# pytest fixtures:
# nwbfile: test.brain_observatory.conftest
# roundtripper: test.brain_observatory.conftest
# running_speed: test.brain_observatory.conftest
# running_acquisition_df_fixture: test.brain_observatory.behavior.conftest


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_acquisition_to_nwbfile(nwbfile, roundtrip, roundtripper,
                                            running_acquisition_df_fixture):
    nwbfile = nwb.add_running_acquisition_to_nwbfile(
        nwbfile, running_acquisition_df_fixture)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    obt_running_acq_df = obt.get_running_acquisition_df()

    pd.testing.assert_frame_equal(running_acquisition_df_fixture,
                                  obt_running_acq_df,
                                  check_like=True)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_speed_to_nwbfile(nwbfile, running_speed,
                                      roundtrip, roundtripper):

    nwbfile = nwb.add_running_speed_to_nwbfile(nwbfile, running_speed)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    obt_running_speed = obt.get_running_speed()

    assert np.allclose(running_speed.timestamps,
                       obt_running_speed['timestamps'])
    assert np.allclose(running_speed.values,
                       obt_running_speed['speed'])


@pytest.mark.parametrize('roundtrip,behavior_stimuli_data_fixture',
                         [(True, {}), (False, {})],
                         indirect=["behavior_stimuli_data_fixture"])
def test_add_stimulus_templates(nwbfile, behavior_stimuli_data_fixture,
                                roundtrip, roundtripper):
    stimulus_templates = get_stimulus_templates(behavior_stimuli_data_fixture,
                                                grating_images_dict={})

    nwb.add_stimulus_template(nwbfile, stimulus_templates)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    stimulus_templates_obt = obt.get_stimulus_templates()

    assert stimulus_templates_obt == stimulus_templates


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_presentations(nwbfile, stimulus_presentations_behavior,
                                    stimulus_timestamps, roundtrip,
                                    roundtripper,
                                    stimulus_templates: StimulusTemplate):
    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)
    nwb.add_stimulus_presentations(nwbfile, stimulus_presentations_behavior)
    nwb.add_stimulus_template(nwbfile=nwbfile,
                              stimulus_template=stimulus_templates)

    # Add index for this template to NWB in-memory object:
    nwb_template = nwbfile.stimulus_template[stimulus_templates.image_set_name]
    compare = (stimulus_presentations_behavior['image_set'] ==
               nwb_template.name)
    curr_stimulus_index = stimulus_presentations_behavior[compare]
    nwb.add_stimulus_index(nwbfile, curr_stimulus_index, nwb_template)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    expected = stimulus_presentations_behavior.copy()
    expected['is_change'] = [False, True, True, True, True]

    obtained = obt.get_stimulus_presentations()

    pd.testing.assert_frame_equal(expected[sorted(expected.columns)],
                                  obtained[sorted(obtained.columns)],
                                  check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_timestamps(nwbfile, stimulus_timestamps,
                                 roundtrip, roundtripper):

    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    np.testing.assert_array_almost_equal(stimulus_timestamps,
                                         obt.get_stimulus_timestamps())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_trials(nwbfile, roundtrip, roundtripper, trials):

    nwb.add_trials(nwbfile, trials, {})

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(trials, obt.get_trials(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_licks(nwbfile, roundtrip, roundtripper, licks):

    nwb.add_licks(nwbfile, licks)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(licks, obt.get_licks(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_rewards(nwbfile, roundtrip, roundtripper, rewards):

    nwb.add_rewards(nwbfile, rewards)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(rewards, obt.get_rewards(),
                                  check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_behavior_only_metadata(roundtrip, roundtripper,
                                    behavior_only_metadata_fixture):

    metadata = behavior_only_metadata_fixture
    nwbfile = pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=metadata['date_of_acquisition']
    )
    nwb.add_metadata(nwbfile, metadata, behavior_only=True)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    metadata_obt = obt.get_metadata()

    assert len(metadata_obt) == len(metadata)
    for key, val in metadata.items():
        assert val == metadata_obt[key]


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_task_parameters(nwbfile, roundtrip,
                             roundtripper, task_parameters):

    nwb.add_task_parameters(nwbfile, task_parameters)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    task_parameters_obt = obt.get_task_parameters()

    assert len(task_parameters_obt) == len(task_parameters)
    for key, val in task_parameters.items():
        if key == 'omitted_flash_fraction':
            if math.isnan(val):
                assert math.isnan(task_parameters_obt[key])
            if math.isnan(task_parameters_obt[key]):
                assert math.isnan(val)
        else:
            assert val == task_parameters_obt[key]


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_task_parameters_stim_nan(nwbfile, roundtrip,
                                      roundtripper,
                                      task_parameters_nan_stimulus_duration):
    """
    Same as test_add_task_parameters, but stimulus_duration_sec is NaN
    """
    task_params = task_parameters_nan_stimulus_duration
    nwb.add_task_parameters(nwbfile, task_params)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    task_parameters_obt = obt.get_task_parameters()

    assert len(task_parameters_obt) == len(task_params)
    for key, val in task_params.items():
        if key in ('omitted_flash_fraction',
                   'stimulus_duration_sec'):
            if math.isnan(val):
                assert math.isnan(task_parameters_obt[key])
            if math.isnan(task_parameters_obt[key]):
                assert math.isnan(val)
        else:
            assert val == task_parameters_obt[key]


def test_write_behavior_nwb():
    """
        This function is intended to test the code in the except block
        of the write_behavior_nwb method. To get there, we simply pass
        None for the session_data parameter which will cause a TypeError
        when the try block attempts to subscript it.

        If this function is ever meaningfully changed in implementation,
        this method of getting to the except block may no longer work and
        this test will need updating.
    """
    with pytest.raises(TypeError) as err:
        write_behavior_nwb(
            session_data=None,
            nwb_filepath=''
        )
    
    assert 'TypeError' in str(err.type)
