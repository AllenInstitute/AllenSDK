import pytest
import os
import datetime
import numpy as np

from pynwb import NWBHDF5IO
from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter

@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_running_speed(nwbfile, tmpfilename, visbeh_pkl):

    test_speed = [0.2697893313797373, 2.294358589025947, 2.294358589025947, 0.26978933137973743, 0.2697893313797375]

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)
    assert np.allclose(visbeh_data.running_speed.data, test_speed)

    nwbfile.add_acquisition(visbeh_data.running_speed)
    with NWBHDF5IO(tmpfilename, mode='w') as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(tmpfilename, mode='r').read()
    assert np.allclose(nwbfile_in.acquisition['running_speed'].data.value, test_speed)


@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_epoch(nwbfile, tmpfilename, visbeh_pkl):

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)

    epoch_table = visbeh_data.get_epoch_table()


    


