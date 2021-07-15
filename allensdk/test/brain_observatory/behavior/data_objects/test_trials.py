from datetime import datetime
from pathlib import Path

import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile, \
    SyncFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.data_objects.trials.trials import \
    Trials
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromStimulusFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_session_id = 994174745
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        expected = pd.read_pickle(str(test_data_dir / 'trials.pkl'))
        cls.expected = Trials(trials=expected)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file = StimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file)
        licks = Licks.from_stimulus_file(stimulus_file=stimulus_file)
        rewards = Rewards.from_stimulus_file(stimulus_file=stimulus_file)
        sync_file = SyncFile.from_lims(
            db=self.dbconn, ophys_experiment_id=self.ophys_experiment_id)
        equipment = Equipment.from_lims(
            behavior_session_id=self.behavior_session_id, lims_db=self.dbconn)
        trials = Trials.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            licks=licks,
            rewards=rewards,
            sync_file=sync_file,
            equipment=equipment
        )
        assert trials == self.expected


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        trials = pd.read_pickle(str(test_data_dir / 'trials.pkl'))
        cls.trials = Trials(trials=trials)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.trials.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Trials)
        else:
            obt = self.trials.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.trials
