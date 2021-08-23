import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromStimulusFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_session_id = 994174745

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        expected = pd.read_pickle(str(test_data_dir / 'rewards.pkl'))
        cls.expected = Rewards(rewards=expected)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file = StimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file)
        rewards = Rewards.from_stimulus_file(stimulus_file=stimulus_file,
                                             stimulus_timestamps=timestamps)
        assert rewards == self.expected

    def test_from_stimulus_file2(self, tmpdir):
        """
        Test that Rewards.from_stimulus_file returns
        expected results (main nuance is that timestamps should be
        determined by applying the reward frame as an index to
        stimulus_timestamps)
        """

        def _create_dummy_stimulus_file():
            trial_log = [
                {'rewards': [(0.001, -1.0, 4)],
                 'trial_params': {'auto_reward': True}},
                {'rewards': []},
                {'rewards': [(0.002, -1.0, 10)],
                 'trial_params': {'auto_reward': False}}
            ]
            data = {
                'items': {
                    'behavior': {
                        'trial_log': trial_log
                    }
                },
            }
            tmp_path = tmpdir / 'stimulus_file.pkl'
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
                f.seek(0)

            return tmp_path

        stimulus_filepath = _create_dummy_stimulus_file()
        stimulus_file = StimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(stimulus_filepath)})
        timestamps = StimulusTimestamps(timestamps=np.arange(0, 2.0, 0.01))
        rewards = Rewards.from_stimulus_file(stimulus_file=stimulus_file,
                                             stimulus_timestamps=timestamps)

        expected_dict = {'volume': [0.001, 0.002],
                         'timestamps': [0.04, 0.1],
                         'autorewarded': [True, False]}
        expected_df = pd.DataFrame(expected_dict)
        expected_df = expected_df
        assert expected_df.equals(rewards.value)


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        rewards = pd.read_pickle(str(test_data_dir / 'rewards.pkl'))
        cls.rewards = Rewards(rewards=rewards)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.rewards.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Rewards)
        else:
            obt = self.rewards.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.rewards
