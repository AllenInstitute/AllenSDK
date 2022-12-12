import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.stimuli\
    .fingerprint_stimulus import \
    FingerprintStimulus
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import get_spontaneous_block_indices, Presentations
from allensdk.internal.brain_observatory.mouse import Mouse


@pytest.mark.parametrize('stimulus_blocks, expected', [
    ([0, 2, 3], [1]),
    ([0, 2, 4], [1, 3]),
    ([0, 1, 2], [])
])
def test_get_spontaneous_block_indices(stimulus_blocks, expected):
    stimulus_blocks = np.array(stimulus_blocks, dtype='int')
    expected = np.array(expected, dtype='int')
    obtained = get_spontaneous_block_indices(
        stimulus_blocks=stimulus_blocks)
    assert np.array_equal(obtained, expected)


class TestFingerprintStimulus:
    @classmethod
    def setup_class(cls):
        stim_file = {
            'items': {
                'behavior': {
                    'items': {
                        'fingerprint': {
                            'static_stimulus': {
                                'runs': 2,
                                # Movie is 2 frames long, repeats 2 times
                                'sweep_frames': np.array([(0, 1), (2, 3),
                                                          (4, 5), (6, 7)]),
                                # 2 frames of gray screen followed by 2 movie
                                # frames
                                'frame_list': np.array([-1, -1, 0, 1])
                            },
                            # 2 gray screen on frame 5, 6 followed by 4 frames
                            # of movie that last 2 monitor frames each
                            'frame_indices': np.array([5, 6] +
                                                      list(range(
                                                          7, 7 + 4 * 2)))
                        }
                    }
                }
            }
        }
        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        with open(Path(tmpdir.name) / 'behavior_stimulus.pkl', 'wb') as f:
            pickle.dump(stim_file, f)
        cls.stimulus_file = BehaviorStimulusFile(
            filepath=Path(tmpdir.name) / 'behavior_stimulus.pkl'
        )
        cls.stimulus_presentations_table = pd.DataFrame({
            'stimulus_block': [0]
        })
        cls.stimulus_timestamps = StimulusTimestamps(
            timestamps=np.arange(0, 20),
            monitor_delay=0.0
        )
        dir = Path(__file__).parent.resolve()
        cls.test_data_dir = dir / 'test_data'

    def teardown(self):
        self.tmpdir.cleanup()

    def test_fingerprint_stimulus(self):
        """Sanity check to make sure fingerprint records are as expected"""
        obt = FingerprintStimulus.from_stimulus_file(
            stimulus_presentations=self.stimulus_presentations_table,
            stimulus_file=self.stimulus_file,
            stimulus_timestamps=self.stimulus_timestamps

        )
        with open(self.test_data_dir / 'fingerprint_stimulus.pkl', 'rb') as f:
            expected = pickle.load(f)

        obt = obt.table[sorted([c for c in obt.table])]
        expected = expected[sorted([c for c in expected])]
        pd.testing.assert_frame_equal(obt, expected)

    def test_add_fingerprint_stimulus(self):
        """Checks that fingerprint block and spontaneous block are correctly
        added to table"""
        with open(self.test_data_dir / 'fingerprint_stimulus.pkl', 'rb') as f:
            fingerprint_stim = pickle.load(f)

        obt = Presentations._add_fingerprint_stimulus(
            stimulus_presentations=self.stimulus_presentations_table,
            stimulus_file=self.stimulus_file,
            stimulus_timestamps=self.stimulus_timestamps
        )
        # there should a block for 0, 1 (spontaneous) and 2 (fingerprint)
        assert sorted(obt['stimulus_block'].unique().tolist()) == [0, 1, 2]

        expected_num_spontaneous_rows = 1
        expected_num_fingerprint_rows = fingerprint_stim.shape[0]
        assert obt[obt['stimulus_name'] == 'spontaneous'].shape[0] == \
               expected_num_spontaneous_rows

        assert obt.shape[0] == \
            self.stimulus_presentations_table.shape[0] + \
            expected_num_spontaneous_rows + \
            expected_num_fingerprint_rows


class TestStimulusPresentations:
    @pytest.mark.parametrize('image_names, expected', [
        (['A', 'B'], {'A': False, 'B': False}),
        (['A', 'C'], {'A': False, 'C': True}),
        (['C', 'omitted', np.nan, None], {'C': True})
    ])
    def test_get_is_image_novel(self, image_names, expected):
        with patch.object(Mouse, attribute='from_behavior_session_id',
                          wraps=lambda behavior_session_id: Mouse('1')):
            with patch.object(
                    Mouse, attribute='get_images_shown',
                    wraps=lambda up_to_behavior_session_id: {'A', 'B'}):
                obt = Presentations._get_is_image_novel(
                    image_names=image_names, behavior_session_id=1)
                assert obt == expected

    def test_add_is_image_novel(self):
        stimulus_presentations = pd.DataFrame({
            'image_name': ['A', 'B', 'C', np.nan]})
        is_image_novel = {
            'A': False,
            'B': False,
            'C': True
        }
        with patch.object(Presentations, attribute='_get_is_image_novel',
                          wraps=lambda image_names, behavior_session_id:
                          is_image_novel):
            Presentations._add_is_image_novel(
                stimulus_presentations=stimulus_presentations,
                behavior_session_id=1
            )
            assert (stimulus_presentations['is_image_novel'].tolist() ==
                   list(is_image_novel.values())
                    # due to last stimulus which is not an image
                    + [np.nan]
                    )
