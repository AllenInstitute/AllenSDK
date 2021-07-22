from datetime import datetime
from pathlib import Path

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
