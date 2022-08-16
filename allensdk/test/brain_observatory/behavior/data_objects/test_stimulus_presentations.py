import numpy as np
import pytest

from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import get_spontaneous_block_indices


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
