from datetime import datetime
from pathlib import Path
from typing import Optional

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
from allensdk.brain_observatory.behavior.data_objects.stimuli.util import \
    calculate_monitor_delay
from allensdk.brain_observatory.behavior.data_objects.trials.trial_table \
    import TrialTable
from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromStimulusFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_session_id = 994174745
        cls.ophys_experiment_id = 994278291

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        # Note: trials.pkl must be created from a BehaviorSession,
        # not a BehaviorOphysExperiment. If it is created from
        # a BehaviorOphysExperiment, the stimulus timestamps will be
        # instantiated from a sync file, rather than a stimulus file.
        # The tests expect stimulus_timestamps to be instantiated
        # from a stimulus file.
        expected = pd.read_pickle(str(test_data_dir / 'trials.pkl'))
        cls.expected = TrialTable(trials=expected)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file, stimulus_timestamps, licks, rewards = \
            self._get_trial_table_data()

        trials = TrialTable.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            licks=licks,
            rewards=rewards
        )
        assert trials == self.expected

    def test_from_stimulus_file2(self):
        dir = Path(__file__).parent.parent.resolve()
        stimulus_filepath = dir / 'resources' / 'example_stimulus.pkl.gz'
        stimulus_file = StimulusFile(filepath=stimulus_filepath)
        stimulus_file, stimulus_timestamps, licks, rewards = \
            self._get_trial_table_data(stimulus_file=stimulus_file)
        TrialTable.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            licks=licks,
            rewards=rewards
        )

    def _get_trial_table_data(self,
                              stimulus_file: Optional[StimulusFile] = None):
        """returns data required to instantiate a TrialTable"""
        if stimulus_file is None:
            stimulus_file = StimulusFile.from_lims(
                behavior_session_id=self.behavior_session_id, db=self.dbconn)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file,
            monitor_delay=0.02115)
        licks = Licks.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        rewards = Rewards.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        return stimulus_file, stimulus_timestamps, licks, rewards


class TestMonitorDelay:
    @classmethod
    def setup_class(cls):
        cls.lookup_table_expected_values = {
            'CAM2P.1': 0.020842,
            'CAM2P.2': 0.037566,
            'CAM2P.3': 0.021390,
            'CAM2P.4': 0.021102,
            'CAM2P.5': 0.021192,
            'MESO.1': 0.03613
        }

    def setup_method(self, method):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        trials = pd.read_pickle(str(test_data_dir / 'trials.pkl'))
        self.sync_file = SyncFile(filepath=str(test_data_dir / 'sync.h5'))
        self.trials = TrialTable(trials=trials)

    @pytest.mark.parametrize('equipment_name', ('CAMP2.1', 'MESO.2'))
    def test_monitor_delay(self, monkeypatch, equipment_name):
        equipment = Equipment(equipment_name=equipment_name)

        def dummy_delay(self):
            return 1.12

        with monkeypatch.context() as ctx:
            ctx.setattr(OphysTimeAligner,
                        '_get_monitor_delay',
                        dummy_delay)
            md = calculate_monitor_delay(sync_file=self.sync_file,
                                         equipment=equipment)
            assert abs(md - 1.12) < 1.0e-6

    def test_monitor_delay_lookup(self, monkeypatch):
        def dummy_delay(self):
            """force monitor delay calculation to fail"""
            raise ValueError("that did not work")

        with monkeypatch.context() as ctx:
            ctx.setattr(OphysTimeAligner,
                        '_get_monitor_delay',
                        dummy_delay)
            for equipment, expected in \
                    self.lookup_table_expected_values.items():
                equipment = Equipment(equipment_name=equipment)
                md = calculate_monitor_delay(
                    sync_file=self.sync_file, equipment=equipment)
                assert abs(md - expected) < 1e-6

    def test_unkown_rig_name(self, monkeypatch):
        def dummy_delay(self):
            """force monitor delay calculation to fail"""
            raise ValueError("that did not work")

        with monkeypatch.context() as ctx:
            ctx.setattr(OphysTimeAligner,
                        '_get_monitor_delay',
                        dummy_delay)
            equipment = Equipment(equipment_name='spam')
            with pytest.raises(RuntimeError):
                calculate_monitor_delay(sync_file=self.sync_file,
                                        equipment=equipment)


class TestNWB:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        trials = pd.read_pickle(str(test_data_dir / 'trials.pkl'))
        cls.trials = TrialTable(trials=trials)

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
                data_object_cls=TrialTable)
        else:
            obt = self.trials.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.trials
