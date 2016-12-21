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

import matplotlib
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
matplotlib.use('agg')
import pytest
from mock import patch, MagicMock
import itertools as it


@pytest.fixture
def dataset():
    dataset = MagicMock(name='dataset')
    
    timestamps = MagicMock(name='timestamps')
    celltraces = MagicMock(name='celltraces')
    dataset.get_corrected_fluorescence_traces = \
        MagicMock(name='get_corrected_fluorescence_traces',
                  return_value=(timestamps, celltraces))
    dataset.get_roi_ids = MagicMock(name='get_roi_ids')
    dataset.get_cell_specimen_ids = MagicMock(name='get_cell_specimen_ids')
    dff_traces = MagicMock(name="dfftraces")
    dataset.get_dff_traces = MagicMock(name='get_dff_traces',
                                       return_value=(None, dff_traces))
    dxcm = MagicMock(name='dxcm')
    dxtime = MagicMock(name='dxtime')
    dataset.get_running_speed=MagicMock(name='get_running_speed',
                                        return_value=(dxcm, dxtime))
    return dataset


@pytest.mark.parametrize('speed_tuning,lazy',
                         it.product((False, True),
                                    (False, True)))
def test_example(dataset,
                 speed_tuning,
                 lazy):
    binned_dx_sp = MagicMock(name='binned_dx_sp')
    binned_cells_sp = MagicMock(name='binned_cells_sp')
    binned_dx_vis = MagicMock(name='binned_dx_vis')
    binned_cells_vis = MagicMock(name='binned_cells_vis')
    peak_run = MagicMock(name='peak_run')
    
    with patch('allensdk.brain_observatory.stimulus_analysis.StimulusAnalysis.get_speed_tuning',
               MagicMock(name='get_speed_tuning',
                         return_value=(binned_dx_sp,
                                       binned_cells_sp,
                                       binned_dx_vis,
                                       binned_cells_vis,
                                       peak_run))) as get_speed_tuning:
        sa = StimulusAnalysis(dataset, speed_tuning, lazy)

        if lazy:
            assert sa._timestamps == StimulusAnalysis._LAZY
            assert sa._celltraces == StimulusAnalysis._LAZY
            assert sa._numbercells == StimulusAnalysis._LAZY
            assert sa._roi_id == StimulusAnalysis._LAZY
            assert sa._cell_id == StimulusAnalysis._LAZY
            assert sa._dfftraces == StimulusAnalysis._LAZY
            assert sa._dxcm == StimulusAnalysis._LAZY
            assert sa._dxtime == StimulusAnalysis._LAZY

            # trigger the lazy bits
            print(sa.timestamps)
            print(sa.roi_id)
            print(sa.cell_id)
            print(sa.dfftraces)
            print(sa.dxtime)
    
            if speed_tuning:
                print(sa.binned_dx_sp)
        else:
            assert sa._timestamps is not StimulusAnalysis._LAZY
            assert sa._celltraces is not StimulusAnalysis._LAZY
            assert sa._numbercells is not StimulusAnalysis._LAZY
            assert sa._roi_id is not StimulusAnalysis._LAZY
            assert sa._cell_id is not StimulusAnalysis._LAZY
            assert sa._dfftraces is not StimulusAnalysis._LAZY
            assert sa._dxcm is not StimulusAnalysis._LAZY
            assert sa._dxtime is not StimulusAnalysis._LAZY

    dataset.get_corrected_fluorescence_traces.assert_called_once_with()
    assert sa._timestamps != StimulusAnalysis._LAZY
    assert sa._celltraces != StimulusAnalysis._LAZY
    assert sa._numbercells != StimulusAnalysis._LAZY

    dataset.get_roi_ids.assert_called_once_with()
    assert sa._roi_id != StimulusAnalysis._LAZY

    dataset.get_cell_specimen_ids.assert_called_once_with()
    assert sa._cell_id != StimulusAnalysis._LAZY

    dataset.get_dff_traces.assert_called_once_with()
    assert sa._dfftraces != StimulusAnalysis._LAZY

    dataset.get_running_speed.assert_called_once_with()
    assert sa._dxcm != StimulusAnalysis._LAZY
    assert sa._dxtime != StimulusAnalysis._LAZY

    if speed_tuning:
        get_speed_tuning.assert_called_once_with(binsize=800)
        assert sa._binned_dx_sp != StimulusAnalysis._LAZY
        assert sa._binned_cells_sp != StimulusAnalysis._LAZY
        assert sa._binned_dx_vis != StimulusAnalysis._LAZY
        assert sa._binned_cells_vis != StimulusAnalysis._LAZY
        assert sa._peak_run != StimulusAnalysis._LAZY
    else:
        assert not get_speed_tuning.called
        assert not hasattr(sa, '_binned_dx_sp')
        assert not hasattr(sa, '_binned_cells_sp')
        assert not hasattr(sa, '_binned_dx_vis')
        assert not hasattr(sa, '_binned_cells_vis')
        assert not hasattr(sa, '_peak_run')

