import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.task_parameters import \
    TaskParameters
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestFromStimulusFile(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_session_id = 994174745

        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        with open(test_data_dir / 'task_parameters.json') as f:
            tp = json.load(f)
        cls.expected = TaskParameters(**tp)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file = StimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        tp = TaskParameters.from_stimulus_file(stimulus_file=stimulus_file)
        assert tp == self.expected


class TestNWB:
    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

        dir = Path(__file__).parent.resolve()
        self.test_data_dir = dir / 'test_data'

        with open(self.test_data_dir / 'task_parameters.json') as f:
            tp = json.load(f)
        self.task_parameters = TaskParameters(**tp)

    @pytest.mark.parametrize('is_stimulus_duration_sec_nan', [True, False])
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture,
                            is_stimulus_duration_sec_nan):
        if is_stimulus_duration_sec_nan:
            self.task_parameters._stimulus_duration_sec = np.nan

        self.task_parameters.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=TaskParameters)
        else:
            obt = TaskParameters.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.task_parameters
