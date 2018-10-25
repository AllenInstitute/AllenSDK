from allensdk.experimental.nwb.stimulus import (VisualBehaviorStimulusAdapter,
                                                VisualCodingStimulusAdapter)
from pynwb import NWBFile, NWBHDF5IO
import datetime
import os
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visual_behavior_running_speed(nwb_filename, vb_pkl, vb_sync):
    adapter = VisualBehaviorStimulusAdapter(vb_pkl, vb_sync, stim_key=2)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    running_speed = adapter.running_speed
    nwbfile.add_acquisition(running_speed)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()
    running_speed_in = nwbfile_in.get_acquisition('running_speed')
    assert(np.allclose(running_speed_in.data, running_speed.data.data))
    assert(np.allclose(running_speed_in.timestamps, running_speed.timestamps))


@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visual_coding_running_speed(nwb_filename, vc_pkl, vc_sync):
    adapter = VisualCodingStimulusAdapter(vc_pkl, vc_sync)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test vc',
        identifier='vc_session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    running_speed = adapter.running_speed
    nwbfile.add_acquisition(running_speed)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()
    running_speed_in = nwbfile_in.get_acquisition('running_speed')
    assert(np.allclose(running_speed_in.data, running_speed.data.data))
    assert(np.allclose(running_speed_in.timestamps, running_speed.timestamps))
