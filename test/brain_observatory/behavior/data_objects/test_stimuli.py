from datetime import datetime
from pathlib import Path

import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations as StimulusPresentations
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import \
    Stimuli
from allensdk.brain_observatory.behavior.data_objects.stimuli.templates \
    import \
    Templates
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromStimulusFile(LimsTest):
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
        stimulus_file = StimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file)
        stimuli = Stimuli.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            limit_to_images=['im065'])
        assert stimuli.presentations == self.expected_presentations
        assert stimuli.templates == self.expected_templates


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
        bsf = StimulusFile(
            filepath=self.test_data_dir / 'behavior_stimulus_file.pkl')
        ts = StimulusTimestamps.from_stimulus_file(stimulus_file=bsf)
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

        # is_change different due to limit_to_images
        obt.presentations.value.drop('is_change', axis=1, inplace=True)

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
