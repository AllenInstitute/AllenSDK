# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from pkg_resources import resource_filename  # @UnresolvedImport
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet, si
import pytest
import os


NWB_FLAVORS = []


if 'TEST_NWB_FILES' in os.environ:
    nwb_list_file = os.environ['TEST_NWB_FILES']
else:
    nwb_list_file = resource_filename(__name__, 'nwb_files.txt')

if nwb_list_file == 'skip':
    NWB_FLAVORS = []
else:
    with open(nwb_list_file, 'r') as f:
        NWB_FLAVORS = [l.strip() for l in f]


@pytest.fixture(params=NWB_FLAVORS)
def data_set(request):
    data_set = BrainObservatoryNwbDataSet(request.param)

    return data_set


def test_acceptance(data_set):
    data_set.get_cell_specimen_ids()
    data_set.get_session_type()
    data_set.get_metadata()
    data_set.get_running_speed()
    data_set.get_motion_correction()


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_roi_ids(data_set):
    ids = data_set.get_roi_ids()
    assert len(ids) == len(data_set.get_cell_specimen_ids())

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_metadata(data_set):
    md = data_set.get_metadata()

    valid_fields = [ 'genotype', 'cre_line', 'imaging_depth_um', 'ophys_experiment_id', 'experiment_container_id',
                     'session_start_time', 'age_days', 'device', 'device_name', 'pipeline_version', 'sex',
                     'targeted_structure', 'excitation_lambda', 'indicator', 'fov', 'session_type', 'specimen_name' ]

    invalid_fields = [ 'imaging_depth', 'age', 'device_string', 'generated_by' ]

    for field in valid_fields:
        assert md[field] is not None

    for field in invalid_fields:
        assert field not in md
    


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_cell_specimen_indices(data_set):
    inds = data_set.get_cell_specimen_indices([])
    assert len(inds) == 0

    ids = data_set.get_cell_specimen_ids()

    inds = data_set.get_cell_specimen_indices(ids)
    assert np.all(np.array(inds) == np.arange(len(inds)))

    inds = data_set.get_cell_specimen_indices([ids[0]])
    assert inds[0] == 0


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_fluorescence_traces(data_set):
    ids = data_set.get_cell_specimen_ids()

    timestamps, traces = data_set.get_fluorescence_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_fluorescence_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_fluorescence_traces([ids[0]])
    assert traces.shape[0] == 1


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_neuropil_traces(data_set):
    ids = data_set.get_cell_specimen_ids()

    timestamps, traces = data_set.get_neuropil_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_neuropil_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_neuropil_traces([ids[0]])
    assert traces.shape[0] == 1


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_dff_traces(data_set):
    ids = data_set.get_cell_specimen_ids()

    timestamps, traces = data_set.get_dff_traces()
    # assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_dff_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_dff_traces([ids[0]])
    assert traces.shape[0] == 1

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_neuropil_r(data_set):

    ids = data_set.get_cell_specimen_ids()
    r = data_set.get_neuropil_r()
    assert len(ids) == len(r)

    r = data_set.get_neuropil_r(ids)
    assert len(ids) == len(r)

    short_list = [ids[0]]
    r = data_set.get_neuropil_r(short_list)
    assert len(short_list) == len(r)

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_corrected_fluorescence_traces(data_set):
    ids = data_set.get_cell_specimen_ids()

    timestamps, traces = data_set.get_corrected_fluorescence_traces()
    assert len(timestamps) == traces.shape[1]
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_corrected_fluorescence_traces(ids)
    assert len(ids) == traces.shape[0]

    timestamps, traces = data_set.get_corrected_fluorescence_traces([ids[0]])
    assert traces.shape[0] == 1


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_roi_mask(data_set):
    ids = data_set.get_cell_specimen_ids()
    roi_masks = data_set.get_roi_mask()
    assert len(ids) == len(roi_masks)

    max_projection = data_set.get_max_projection()
    for roi_mask in roi_masks:
        mask = roi_mask.get_mask_plane()
        assert mask.shape[0] == max_projection.shape[0]
        assert mask.shape[1] == max_projection.shape[1]

    roi_masks = data_set.get_roi_mask([ids[0]])
    assert len(roi_masks) == 1


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_roi_mask_array(data_set):
    ids = data_set.get_cell_specimen_ids()
    arr = data_set.get_roi_mask_array()
    assert arr.shape[0] == len(ids)

    arr = data_set.get_roi_mask_array([ids[0]])
    assert arr.shape[0] == 1

    try:
        arr = data_set.get_roi_mask_array([0])
    except ValueError as e:
        assert str(e).startswith("Cell specimen not found")


@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_stimulus_epoch_table(data_set):

    summary_df = data_set.get_stimulus_epoch_table()

    session_type = data_set.get_session_type()
    if session_type == si.THREE_SESSION_A or si.THREE_SESSION_C:
        assert len(summary_df) == 7
    elif session_type == si.THREE_SESSION_B:
        assert len(summary_df) == 8
    elif session_type == si.THREE_SESSION_C2:
        assert len(summary_df) == 10
    else:
        raise NotImplementedError('Code not tested for session of type: %s' % session_type)

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_get_stimulus_table_master(data_set):

    master_df = data_set.get_stimulus_table('master')

    session_type = data_set.get_session_type()
    if session_type == si.THREE_SESSION_A:
        assert len(master_df) == 45629
    elif session_type == si.THREE_SESSION_B:
        assert len(master_df) == 20951
    elif session_type == si.THREE_SESSION_C:
        assert len(master_df) == 26882
    elif session_type == si.THREE_SESSION_C2:
        assert len(master_df) == 29398
    else:
        raise NotImplementedError('Code not tested for session of type: %s' % session_type)
