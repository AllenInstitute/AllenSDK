import numpy as np
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet

TEST_FILE = '/projects/neuralcoding/vol1/prod6/specimen_497258322/ophys_experiment_506954308/506954308.nwb'

def test_acceptance():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()
    ds.get_session_type()
    ds.get_metadata()
    ds.get_running_speed()
    ds.get_motion_correction()

def test_get_roi_ids():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)
    
    ids = ds.get_roi_ids()
    assert len(ids) == len(ds.get_cell_specimen_ids())

def test_get_cell_specimen_indices():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)
    
    inds = ds.get_cell_specimen_indices([])
    assert len(inds) == 0

    ids = ds.get_cell_specimen_ids()
    
    inds = ds.get_cell_specimen_indices(ids)
    assert np.all(np.array(inds) == np.arange(len(inds)))

    inds = ds.get_cell_specimen_indices([ids[0]])
    assert inds[0] == 0

def test_get_fluorescence_traces():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()

    timestamps, traces = ds.get_fluorescence_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_fluorescence_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_fluorescence_traces([ids[0]])
    assert traces.shape[0] == 1

def test_get_neuropil_traces():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()

    timestamps, traces = ds.get_neuropil_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_neuropil_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_neuropil_traces([ids[0]])
    assert traces.shape[0] == 1

def test_get_dff_traces():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()

    timestamps, traces = ds.get_dff_traces()
    #assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_dff_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_dff_traces([ids[0]])
    assert traces.shape[0] == 1

def test_get_corrected_fluorescence_traces():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()

    timestamps, traces = ds.get_corrected_fluorescence_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_corrected_fluorescence_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = ds.get_corrected_fluorescence_traces([ids[0]])
    assert traces.shape[0] == 1

def test_get_roi_mask():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()
    roi_masks = ds.get_roi_mask()
    assert len(ids) == len(roi_masks)

    for roi_mask in roi_masks:
        m = roi_mask.get_mask_plane()
        assert m.shape[0] == ds.MOVIE_FOV_PX[0]
        assert m.shape[1] == ds.MOVIE_FOV_PX[1]

    
    roi_masks = ds.get_roi_mask([ids[0]])
    assert len(roi_masks) == 1

def test_get_roi_mask_array():
    ds = BrainObservatoryNwbDataSet(TEST_FILE)

    ids = ds.get_cell_specimen_ids()
    arr = ds.get_roi_mask_array()
    assert arr.shape[0] == len(ids)

    arr = ds.get_roi_mask_array([ids[0]])
    assert arr.shape[0] == 1
    
    try:
        arr = ds.get_roi_mask_array([0])
    except ValueError as e:
        assert str(e).startswith("Cell specimen not found")

