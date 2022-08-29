import datetime
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.internal.brain_observatory.mouse import Mouse


class TestMouse:
    @classmethod
    def setup_class(cls):
        cls.mouse = Mouse(mouse_id='1')
        cls.image_names = ('A', 'B')
        cls.tmpdir = tempfile.TemporaryDirectory()

    def teardown_class(self):
        self.tmpdir.cleanup()

    @staticmethod
    def get_behavior_sessions():
        """Returns behavior sessions for this mouse.
        Varies the date of acquisition. behavior session id is same as day
        """
        return [
            BehaviorMetadata(
                date_of_acquisition=DateOfAcquisition(
                    date_of_acquisition=datetime.datetime(
                        year=2022, month=12, day=day)),
                behavior_session_id=BehaviorSessionId(behavior_session_id=day),
                behavior_session_uuid=None,
                equipment=None,
                session_type=None,
                stimulus_frame_rate=None,
                subject_metadata=None
            )
            for day in range(1, 11)]

    def get_behavior_stimulus_file(self, behavior_session_id, db):
        # need to create 10 dummy stimulus dictionaries for each of the 10
        # behavior sessions for this mouse
        behavior_session_id = int(behavior_session_id)
        if behavior_session_id <= 5:
            stimulus_category = 'image'
            if behavior_session_id % 2 == 0:
                image_name = self.image_names[0]
            else:
                image_name = self.image_names[1]
        else:
            stimulus_category = 'grating'
            image_name = None
        d = {
            'items': {
                'behavior': {
                    'stimuli': {
                        # unused
                        '': {
                            'set_log': [
                                (stimulus_category,
                                 image_name, '', '') for _ in range(10)]
                        }
                    }
                }
            }
        }
        with open(Path(self.tmpdir.name) / f'stim_{behavior_session_id}.pkl',
                  'wb') as f:
            pickle.dump(d, f)
        return BehaviorStimulusFile(
            filepath=Path(self.tmpdir.name) / f'stim_{behavior_session_id}'
                                              f'.pkl')

    @pytest.mark.parametrize('upto_behavior_session_id', (None, 1, 2, 6, 9))
    def test_images_shown(self, upto_behavior_session_id):
        with patch(
                'allensdk.internal.brain_observatory.mouse.'
                'db_connection_creator',
                wraps=lambda fallback_credentials: None):
            with patch.object(Mouse, attribute='get_behavior_sessions',
                              wraps=self.get_behavior_sessions):
                with patch.object(BehaviorStimulusFile, attribute='from_lims',
                                  wraps=self.get_behavior_stimulus_file):
                    obt = self.mouse.get_images_shown(
                        up_to_behavior_session_id=upto_behavior_session_id,
                        n_workers=1)
                    if upto_behavior_session_id is None:
                        assert obt == set(self.image_names)
                    elif upto_behavior_session_id == 1:
                        # hasn't been shown any images previously
                        assert obt == set()
                    elif upto_behavior_session_id == 2:
                        # only been shown B
                        assert obt == {'B'}
                    elif upto_behavior_session_id == 6:
                        # shown all images
                        assert obt == set(self.image_names)
                    elif upto_behavior_session_id == 9:
                        # shown all images
                        assert obt == set(self.image_names)
