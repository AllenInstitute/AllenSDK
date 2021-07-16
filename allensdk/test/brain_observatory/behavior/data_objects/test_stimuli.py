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
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import \
    Stimuli
from allensdk.brain_observatory.behavior.data_objects.stimuli.templates \
    import \
    Templates
from allensdk.brain_observatory.behavior.data_objects.trials.trials import \
    Trials
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
        cls.expected_presentations = Presentations(presentations=presentations)
        cls.expected_templates = Templates(templates=templates)

    @pytest.mark.requires_bamboo
    def test_from_stimulus_file(self):
        stimulus_file = StimulusFile.from_lims(
            behavior_session_id=self.behavior_session_id, db=self.dbconn)
        stimuli = Stimuli.from_stimulus_file(stimulus_file=stimulus_file,
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
        p = Presentations(presentations=presentations)
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
