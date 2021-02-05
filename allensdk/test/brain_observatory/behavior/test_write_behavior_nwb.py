import math

import numpy as np
import pandas as pd
import pynwb
import pytest

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorNwbApi)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_speed_to_nwbfile(nwbfile, running_speed,
                                      roundtrip, roundtripper):

    nwbfile = nwb.add_running_speed_to_nwbfile(nwbfile, running_speed)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    running_speed_obt = obt.get_running_speed()
    assert np.allclose(running_speed.timestamps, running_speed_obt.timestamps)
    assert np.allclose(running_speed.values, running_speed_obt.values)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_data_dfs_to_nwbfile(nwbfile, running_data_df,
                                         roundtrip, roundtripper):
    running_data_df_unfiltered = running_data_df.copy()
    running_data_df_unfiltered['speed'] = running_data_df['speed'] * 2

    unit_dict = {'v_sig': 'V', 'v_in': 'V',
                 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
    nwbfile = nwb.add_running_data_dfs_to_nwbfile(nwbfile,
                                                  running_data_df,
                                                  running_data_df_unfiltered,
                                                  unit_dict)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(
        running_data_df, obt.get_running_data_df(lowpass=True))
    pd.testing.assert_frame_equal(
        running_data_df_unfiltered, obt.get_running_data_df(lowpass=False))


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_templates(nwbfile, stimulus_templates,
                                roundtrip, roundtripper):
    for key, val in stimulus_templates.items():
        nwb.add_stimulus_template(nwbfile, val, key)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    stimulus_templates_obt = obt.get_stimulus_templates()
    template_union = (
        set(stimulus_templates.keys()) | set(stimulus_templates_obt.keys()))
    for key in template_union:
        np.testing.assert_array_almost_equal(stimulus_templates[key],
                                             stimulus_templates_obt[key])


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_presentations(nwbfile, stimulus_presentations_behavior,
                                    stimulus_timestamps, roundtrip,
                                    roundtripper, stimulus_templates):
    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)
    nwb.add_stimulus_presentations(nwbfile, stimulus_presentations_behavior)
    for key, val in stimulus_templates.items():
        nwb.add_stimulus_template(nwbfile, val, key)

        # Add index for this template to NWB in-memory object:
        nwb_template = nwbfile.stimulus_template[key]
        curr_stimulus_index = stimulus_presentations_behavior[
            stimulus_presentations_behavior['image_set'] == nwb_template.name]
        nwb.add_stimulus_index(nwbfile, curr_stimulus_index, nwb_template)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorNwbApi)
    else:
        obt = BehaviorNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(stimulus_presentations_behavior,
                                  obt.get_stimulus_presentations(),
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
                                    behavior_only_metadata):

    metadata = behavior_only_metadata
    nwbfile = pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=metadata['experiment_datetime']
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
