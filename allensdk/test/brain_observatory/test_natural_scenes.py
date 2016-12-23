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
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
matplotlib.use('agg')
import pytest
from mock import patch, MagicMock


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
    dataset.get_stimulus_table=MagicMock(name='get_stimulus_table',
                                         return_value=MagicMock())
    
    return dataset

def mock_speed_tuning():
    binned_dx_sp = MagicMock(name='binned_dx_sp')
    binned_cells_sp = MagicMock(name='binned_cells_sp')
    binned_dx_vis = MagicMock(name='binned_dx_vis')
    binned_cells_vis = MagicMock(name='binned_cells_vis')
    peak_run = MagicMock(name='peak_run')
    
    return MagicMock(name='get_speed_tuning',
                     return_value=(binned_dx_sp,
                                   binned_cells_sp,
                                   binned_dx_vis,
                                   binned_cells_vis,
                                   peak_run))

def mock_sweep_response():
    sweep_response = MagicMock(name='sweep_response')
    mean_sweep_response = MagicMock(name='mean_sweep_response')
    pval = MagicMock(name='pval')
    
    return MagicMock(name='get_sweep_response',
                     return_value=(sweep_response,
                                   mean_sweep_response,
                                   pval))

@patch.object(StimulusAnalysis,
              'get_speed_tuning',
              mock_speed_tuning())
@patch.object(StimulusAnalysis,
              'get_sweep_response',
              mock_sweep_response())
@pytest.mark.parametrize('trigger', (1, 2, 3, 4, 5))
def test_harness(dataset, trigger):
    ns = NaturalScenes(dataset)

    assert ns._stim_table is StimulusAnalysis._LAZY
    assert ns._number_scenes is StimulusAnalysis._LAZY
    assert ns._sweeplength is StimulusAnalysis._LAZY
    assert ns._interlength is StimulusAnalysis._LAZY
    assert ns._extralength is StimulusAnalysis._LAZY
    assert ns._sweep_response is StimulusAnalysis._LAZY
    assert ns._mean_sweep_response is StimulusAnalysis._LAZY
    assert ns._pval is StimulusAnalysis._LAZY
    assert ns._response is StimulusAnalysis._LAZY
    assert ns._peak is StimulusAnalysis._LAZY

    if trigger == 1:
        print(ns._stim_table)
        print(ns.sweep_response)
        print(ns.response)
        print(ns.peak)
    if trigger == 2:
        print(ns.number_scenes)
        print(ns.sweep_response)
        print(ns.response)
        print(ns.peak)
    elif trigger == 3:
        print(ns.sweeplength)
        print(ns.mean_sweep_response)
        print(ns.response)
        print(ns.peak)
    elif trigger == 4:
        print(ns.interlength)
        print(ns.sweep_response)
        print(ns.response)
        print(ns.peak)
    elif trigger == 5:
        print(ns.extralength)
        print(ns.mean_sweep_response)
        print(ns.response)
        print(ns.peak)

    assert ns._stim_table is not StimulusAnalysis._LAZY
    assert ns._number_scenes is not StimulusAnalysis._LAZY
    assert ns._sweeplength is not StimulusAnalysis._LAZY
    assert ns._interlength is not StimulusAnalysis._LAZY
    assert ns._extralength is not StimulusAnalysis._LAZY
    assert ns._sweep_response is not StimulusAnalysis._LAZY
    assert ns._mean_sweep_response is not StimulusAnalysis._LAZY
    assert ns._pval is not StimulusAnalysis._LAZY
    assert ns._response is not StimulusAnalysis._LAZY
    assert ns._peak is not StimulusAnalysis._LAZY
    # check super properties
    dataset.get_corrected_fluorescence_traces.assert_called_once_with()
    assert ns._timestamps != NaturalScenes._LAZY
    assert ns._celltraces != NaturalScenes._LAZY
    assert ns._numbercells != NaturalScenes._LAZY
    
    dataset.get_roi_ids.assert_called_once_with()
    assert ns._roi_id != NaturalScenes._LAZY
    
    assert dataset.get_cell_specimen_ids.called
    assert ns._cell_id != NaturalScenes._LAZY
    
    dataset.get_dff_traces.assert_called_once_with()
    assert ns._dfftraces != NaturalScenes._LAZY
    
    assert ns._dxcm != NaturalScenes._LAZY
    assert ns._dxtime != NaturalScenes._LAZY
