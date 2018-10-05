import pytest
import os
import datetime
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO
from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter

@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_running_speed(nwbfile, tmpfilename, visbeh_pkl):

    test_speed = [0.26978933, 2.29435859, 4.31892785, 4.31892785, 4.31892785, 4.31892785]

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)
    assert np.allclose(visbeh_data.running_speed.data, test_speed)

    nwbfile.add_acquisition(visbeh_data.running_speed)
    with NWBHDF5IO(tmpfilename, mode='w') as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(tmpfilename, mode='r').read()
    assert np.allclose(nwbfile_in.acquisition['running_speed'].data.value, test_speed)


@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_epoch(nwbfile, tmpfilename, visbeh_pkl):
    tmpfilename = '/home/nicholasc/foo.nwb'
    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)

    epoch_table = visbeh_data.get_epoch_table()
    nwbfile.epochs = epoch_table

    src_df = nwbfile.epochs.to_dataframe()
    nwbfile.add_acquisition(visbeh_data.running_speed)

    with NWBHDF5IO(tmpfilename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(tmpfilename, mode='r').read()

    tgt_df = nwbfile_in.epochs.to_dataframe()

    # Assert the round-trip worked
    ts_src = src_df['timeseries']
    ts_tgt = tgt_df['timeseries']
    src_df.drop('timeseries', axis=1, inplace=True)
    tgt_df.drop('timeseries', axis=1, inplace=True)

    for tgt_row, src_row in zip(ts_tgt.values, ts_src.values):
        for tgt_ts, src_ts in zip(tgt_row, src_row):
            assert np.allclose(tgt_ts.data, src_ts.data)
            assert np.allclose(tgt_ts.timestamps, src_ts.timestamps)

    pd.testing.assert_frame_equal(src_df, tgt_df)
