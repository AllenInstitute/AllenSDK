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
from allensdk.brain_observatory.natural_movie import NaturalMovie
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
import pytest
from mock import patch, MagicMock
import pandas as pd


@pytest.fixture
def stimulus_table():
    return pd.DataFrame([
        {'frame': 0, 'start': 0, 'stop': 1},
        {'frame': 0, 'start': 1, 'stop': 2},
        {'frame': 1, 'start': 2, 'stop': 3},
    ])


@pytest.fixture
def dataset(stimulus_table):
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
                                         return_value=stimulus_table)
    
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
@pytest.mark.parametrize(
    'trigger', [
        ('stim_table', 'sweep_response', 'peak'),
        ('sweeplength', 'sweep_response', 'peak')
    ]
)
def test_harness(dataset, trigger):
    movie_name = "Mock Movie Name"
    nm = NaturalMovie(dataset, movie_name)

    assert nm._stim_table is StimulusAnalysis._PRELOAD
    assert nm._sweeplength is StimulusAnalysis._PRELOAD
    assert nm._sweep_response is StimulusAnalysis._PRELOAD
    assert nm._peak is StimulusAnalysis._PRELOAD

    for attr in trigger:
        print(getattr(nm, attr))

    assert nm._stim_table is not StimulusAnalysis._PRELOAD
    assert nm._sweeplength is not StimulusAnalysis._PRELOAD
    assert nm._sweep_response is not StimulusAnalysis._PRELOAD
    assert nm._peak is not StimulusAnalysis._PRELOAD

    # check super properties weren't preloaded
    dataset.get_corrected_fluorescence_traces.assert_called_once_with()
    assert nm._timestamps is not NaturalMovie._PRELOAD
    assert nm._celltraces is not NaturalMovie._PRELOAD
    assert nm._numbercells is not NaturalMovie._PRELOAD

    assert not dataset.get_roi_ids.called
    assert nm._roi_id is NaturalMovie._PRELOAD

    assert dataset.get_cell_specimen_ids.called
    assert nm._cell_id is NaturalMovie._PRELOAD

    assert not dataset.get_dff_traces.called
    assert nm._dfftraces is NaturalMovie._PRELOAD

    assert nm._dxcm is NaturalMovie._PRELOAD
    assert nm._dxtime is NaturalMovie._PRELOAD
