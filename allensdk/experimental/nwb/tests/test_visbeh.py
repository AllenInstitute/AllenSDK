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

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)

    epoch_table = visbeh_data.stimulus_epoch_table
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

@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
def test_visbeh_image_stimulus(nwbfile, tmpfilename, visbeh_pkl):

    visbeh_data = VisualBehaviorStimulusAdapter(visbeh_pkl)

    for x in visbeh_data.image_series_list:
        nwbfile.add_stimulus_template(x)
    
    for y in visbeh_data.index_series_list:
        nwbfile.add_stimulus(y)

    with NWBHDF5IO(tmpfilename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(tmpfilename, mode='r').read()

    assert np.allclose(nwbfile.stimulus['image_index'].data, nwbfile_in.stimulus['image_index'].data)
    assert np.allclose(nwbfile.stimulus['image_index'].timestamps, nwbfile_in.stimulus['image_index'].timestamps)

    assert np.allclose(nwbfile.stimulus['image_index'].indexed_timeseries.data, nwbfile_in.stimulus['image_index'].indexed_timeseries.data)
    assert np.allclose(nwbfile.stimulus['image_index'].indexed_timeseries.timestamps, nwbfile_in.stimulus['image_index'].indexed_timeseries.timestamps)

@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
@pytest.mark.parametrize('pklfile', ['/allen/programs/braintv/production/neuralcoding/prod0/specimen_738720433/behavior_session_760658830/181004091143_409296_2e5b5a55-af4b-4f94-829d-0048df1eb550.pkl',
                                     '/allen/programs/braintv/production/visualbehavior/prod0/specimen_738786518/behavior_session_759866491/181002090744_403468_7add2e7c-96fd-4aa0-b864-3dc4d4c38efa.pkl'])
def test_visbeh_pickle_integration(pklfile, nwbfile, tmpfilename):

    visbeh_data = VisualBehaviorStimulusAdapter(pklfile)

    nwbfile.add_acquisition(visbeh_data.running_speed)
    
    for x in visbeh_data.image_series_list:
        nwbfile.add_stimulus_template(x)
    
    for y in visbeh_data.index_series_list:
        nwbfile.add_stimulus(y)

    with NWBHDF5IO(tmpfilename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(tmpfilename, mode='r').read()