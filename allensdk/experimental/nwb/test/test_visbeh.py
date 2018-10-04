import pytest
import os
import datetime
import numpy as np


from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter

@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_running_speed(nwbfile, tmpfilename, visbeh_pkl):

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)

    assert np.allclose(visbeh_data.running_speed.data, [0.2697893313797373, 2.294358589025947, 2.294358589025947, 0.26978933137973743, 0.2697893313797375])
    


