from datetime import datetime
from pathlib import Path

import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations as StimulusPresentations, Presentations
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import \
    Stimuli
from allensdk.brain_observatory.behavior.data_objects.stimuli.templates \
    import \
    Templates
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromBehaviorStimulusFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_session_id = 994174745

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        presentations = \
            pd.read_pickle(str(test_data_dir / 'presentations.pkl'))
        templates = \
            pd.read_pickle(str(test_data_dir / 'templates.pkl'))
        cls.expected_presentations = StimulusPresentations(
            presentations=presentations)
        cls.expected_templates = Templates(templates=templates)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file = BehaviorStimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file,
            monitor_delay=0.0)
        stimuli = Stimuli.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            limit_to_images=['im065'],
            behavior_session_id=self.behavior_session_id
        )
        assert stimuli.presentations == self.expected_presentations
        assert stimuli.templates == self.expected_templates


@pytest.fixture(scope='module')
def presentations_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return a Presentations object
    """
    obj = Presentations.from_path(
        path=behavior_ecephys_session_config_fixture['stim_table_file'],
        behavior_session_id=(
            behavior_ecephys_session_config_fixture['behavior_session_id'])
    )
    return obj


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip, add_is_change',
                         ([True, False], [True, False]))
def test_read_write_nwb(
        roundtrip,
        add_is_change,
        data_object_roundtrip_fixture,
        presentations_fixture,
        behavior_ecephys_session_config_fixture,
        helper_functions):

    nwbfile = helper_functions.create_blank_nwb_file()

    # Need to write stimulus timestamps first
    bsf = BehaviorStimulusFile.from_json(
        dict_repr=behavior_ecephys_session_config_fixture)
    ts = StimulusTimestamps.from_stimulus_file(stimulus_file=bsf,
                                               monitor_delay=0.0)
    ts.to_nwb(nwbfile=nwbfile)

    presentations_fixture.to_nwb(
        nwbfile=nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=Presentations,
            add_is_change=add_is_change
        )
    else:
        obt = Presentations.from_nwb(
            nwbfile=nwbfile,
            add_is_change=add_is_change)

    assert obt == presentations_fixture


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        cls.test_data_dir = dir / 'test_data'

        presentations = \
            pd.read_pickle(str(cls.test_data_dir / 'presentations.pkl'))
        templates = \
            pd.read_pickle(str(cls.test_data_dir / 'templates.pkl'))
        presentations = presentations.drop('is_change', axis=1)
        presentations = presentations.drop('flashes_since_change', axis=1)
        p = StimulusPresentations(presentations=presentations)
        t = Templates(templates=templates)
        cls.stimuli = Stimuli(presentations=p, templates=t)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

        # Need to write stimulus timestamps first
        bsf = BehaviorStimulusFile(
            filepath=self.test_data_dir / 'behavior_stimulus_file.pkl')
        ts = StimulusTimestamps.from_stimulus_file(stimulus_file=bsf,
                                                   monitor_delay=0.0)
        ts.to_nwb(nwbfile=self.nwbfile)

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.stimuli.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Stimuli)
        else:
            obt = Stimuli.from_nwb(nwbfile=self.nwbfile)

        # is_change different due to limit_to_images. flashes_since_change
        # also relies on this column so we ommit that.
        obt.presentations.value.drop('is_change', axis=1, inplace=True)
        obt.presentations.value.drop('flashes_since_change',
                                     axis=1,
                                     inplace=True)

        assert obt == self.stimuli


@pytest.mark.parametrize("stimulus_table, expected_table_data", [
    ({'image_index': [8, 9],
      'image_name': ['omitted', 'not_omitted'],
      'image_set': ['omitted', 'not_omitted'],
      'index': [201, 202],
      'omitted': [True, False],
      'start_frame': [231060, 232340],
      'start_time': [0, 250],
      'stop_time': [None, 1340509],
      'duration': [None, 1340259]},
     {'image_index': [8, 9],
      'image_name': ['omitted', 'not_omitted'],
      'image_set': ['omitted', 'not_omitted'],
      'index': [201, 202],
      'omitted': [True, False],
      'start_frame': [231060, 232340],
      'start_time': [0, 250],
      'stop_time': [0.25, 1340509],
      'duration': [0.25, 1340259]}
     )
])
def test_set_omitted_stop_time(stimulus_table, expected_table_data):
    stimulus_table = pd.DataFrame.from_dict(data=stimulus_table)
    expected_table = pd.DataFrame.from_dict(data=expected_table_data)
    stimulus_table = \
        StimulusPresentations._fill_missing_values_for_omitted_flashes(
            df=stimulus_table)
    assert stimulus_table.equals(expected_table)


@pytest.fixture(scope='module')
def stimulus_templates_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return a Templates object
    """

    sf = BehaviorStimulusFile.from_json(
        dict_repr=behavior_ecephys_session_config_fixture)
    obj = Templates.from_stimulus_file(stimulus_file=sf)
    return obj


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_nwb_no_image_index(
        roundtrip,
        data_object_roundtrip_fixture,
        stimulus_templates_fixture,
        presentations_fixture,
        helper_functions):
    """This presentations table has no image_index.
    Make sure the roundtrip doesn't break"""

    nwbfile = helper_functions.create_blank_nwb_file()

    stimulus_templates_fixture.to_nwb(
        nwbfile=nwbfile,
        stimulus_presentations=presentations_fixture)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=Templates
        )
    else:
        obt = Templates.from_nwb(nwbfile=nwbfile)

    assert obt == stimulus_templates_fixture
