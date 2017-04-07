# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from pkg_resources import resource_filename  # @UnresolvedImport
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
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
