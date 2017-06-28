import pytest
import numpy as np
import os
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet, si
from pkg_resources import resource_filename  # @UnresolvedImport
NWB_FLAVORS = []

if 'TEST_NWB_FILES' in os.environ:
    nwb_list_file = os.environ['TEST_NWB_FILES']
else:
    nwb_list_file = resource_filename(__name__, os.path.join('..','core','nwb_files.txt'))

if nwb_list_file == 'skip':
    NWB_FLAVORS = []
else:
    with open(nwb_list_file, 'r') as f:
        NWB_FLAVORS = [l.strip() for l in f]

@pytest.fixture(params=NWB_FLAVORS)
def data_set(request):
    data_set = BrainObservatoryNwbDataSet(request.param)

    return data_set

def test_BinaryIntervalSearchTree():

    bist = si.BinaryIntervalSearchTree([(0, .9, 'A'), (1, 1.9, 'B'), (3, 3.9, 'D'), (2, 2.9, 'C')])
    assert bist.search(1.5)[2] == 'B'
    assert bist.search(0)[2] == 'A'
    assert bist.search(2.5)[2] == 'C'
    assert bist.search(3.5)[2] == 'D'

def test_pixels_to_visual_degrees():
    m = si.BrainObservatoryMonitor()
    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(1), 0.103270443661,10)

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_StimulusSearch(data_set):

    epoch_df = data_set.get_stimulus_epoch_table()
    s = si.StimulusSearch(data_set)
    assert len(s.search(epoch_df.iloc[2]['end'])) == 3
    assert s.search(epoch_df.iloc[2]['end'] + 1) is None

if __name__ == "__main__":

    test_BinaryIntervalSearchTree()
    test_StimulusSearch()