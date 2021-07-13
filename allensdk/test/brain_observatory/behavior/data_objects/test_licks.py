from datetime import datetime
from pathlib import Path

import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.licks import Licks


class TestFromStimulusFile:
    @classmethod
    def setup_class(cls):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'

        cls.stimulus_file = StimulusFile(
            filepath=test_data_dir / 'behavior_stimulus_file.pkl')
        expected = pd.read_pickle(str(test_data_dir / 'licks.pkl'))
        cls.expected = Licks(licks=expected)

    def test_from_stimulus_file(self):
        licks = Licks.from_stimulus_file(stimulus_file=self.stimulus_file)
        assert licks == self.expected


class TestNWB:
    @classmethod
    def setup_class(cls):
        tsf = TestFromStimulusFile()
        tsf.setup_class()
        cls.licks = Licks.from_stimulus_file(
            stimulus_file=tsf.stimulus_file)

    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='1234',
            session_start_time=datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self.licks.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self.nwbfile,
                data_object_cls=Licks)
        else:
            obt = self.licks.from_nwb(nwbfile=self.nwbfile)

        assert obt == self.licks
