# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2016-2017. Allen Institute. All rights reserved.
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
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
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

    assert ns._stim_table is StimulusAnalysis._PRELOAD
    assert ns._number_scenes is StimulusAnalysis._PRELOAD
    assert ns._sweeplength is StimulusAnalysis._PRELOAD
    assert ns._interlength is StimulusAnalysis._PRELOAD
    assert ns._extralength is StimulusAnalysis._PRELOAD
    assert ns._sweep_response is StimulusAnalysis._PRELOAD
    assert ns._mean_sweep_response is StimulusAnalysis._PRELOAD
    assert ns._pval is StimulusAnalysis._PRELOAD
    assert ns._response is StimulusAnalysis._PRELOAD
    assert ns._peak is StimulusAnalysis._PRELOAD

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

    assert ns._stim_table is not StimulusAnalysis._PRELOAD
    assert ns._number_scenes is not StimulusAnalysis._PRELOAD
    assert ns._sweeplength is not StimulusAnalysis._PRELOAD
    assert ns._interlength is not StimulusAnalysis._PRELOAD
    assert ns._extralength is not StimulusAnalysis._PRELOAD
    assert ns._sweep_response is not StimulusAnalysis._PRELOAD
    assert ns._mean_sweep_response is not StimulusAnalysis._PRELOAD
    assert ns._pval is not StimulusAnalysis._PRELOAD
    assert ns._response is not StimulusAnalysis._PRELOAD
    assert ns._peak is not StimulusAnalysis._PRELOAD

    # check super properties
    dataset.get_corrected_fluorescence_traces.assert_called_once_with()
    assert ns._timestamps != NaturalScenes._PRELOAD
    assert ns._celltraces != NaturalScenes._PRELOAD
    assert ns._numbercells != NaturalScenes._PRELOAD

    assert not dataset.get_roi_ids.called
    assert ns._roi_id is NaturalScenes._PRELOAD

    assert dataset.get_cell_specimen_ids.called
    assert ns._cell_id is NaturalScenes._PRELOAD

    assert not dataset.get_dff_traces.called
    assert ns._dfftraces is NaturalScenes._PRELOAD

    assert ns._dxcm is NaturalScenes._PRELOAD
    assert ns._dxtime is NaturalScenes._PRELOAD
