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
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
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

    LSN = MagicMock(name='LSN')
    LSN_mask = MagicMock(name='LSN_mask')
    dataset.get_locally_sparse_noise_stimulus_template = \
        MagicMock(name='get_locally_sparse_noise_stimulus_template',
                  return_value=(LSN, LSN_mask))

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
              'get_sweep_response',
              mock_sweep_response())
@patch.object(LocallySparseNoise,
              'get_receptive_field',
              MagicMock(name='get_receptive_field'))
@pytest.mark.parametrize('stimulus,trigger',
                         it.product(('locally_sparse_noise',
                                     'locally_sparse_noise_4deg',
                                     'locally_sparse_noise_8deg'),
                                    (1,2,3,4,5,6)))
def test_harness(dataset,
                 stimulus,
                 trigger):
    with patch('allensdk.brain_observatory.stimulus_analysis.StimulusAnalysis.get_speed_tuning',
               mock_speed_tuning()) as get_speed_tuning:
        lsn = LocallySparseNoise(dataset, stimulus)

        assert lsn._stim_table is StimulusAnalysis._PRELOAD
        assert lsn._LSN is StimulusAnalysis._PRELOAD
        assert lsn._LSN_mask is StimulusAnalysis._PRELOAD
        assert lsn._sweeplength is StimulusAnalysis._PRELOAD
        assert lsn._interlength is StimulusAnalysis._PRELOAD
        assert lsn._extralength is StimulusAnalysis._PRELOAD
        assert lsn._sweep_response is StimulusAnalysis._PRELOAD
        assert lsn._mean_sweep_response is StimulusAnalysis._PRELOAD
        assert lsn._pval is StimulusAnalysis._PRELOAD
        assert lsn._receptive_field is StimulusAnalysis._PRELOAD
    
        if trigger == 1:
            print(lsn.stim_table)
            print(lsn.sweep_response)
            print(lsn.receptive_field)
        elif trigger == 2:
            print(lsn.LSN)
            print(lsn.mean_sweep_response)
            print(lsn.receptive_field)
        elif trigger == 3:
            print(lsn.LSN_mask)
            print(lsn.pval)
            print(lsn.receptive_field)
        elif trigger == 4:
            print(lsn.sweeplength)
            print(lsn.sweep_response)
            print(lsn.receptive_field)
        elif trigger == 5:
            print(lsn.interlength)
            print(lsn.mean_sweep_response)
            print(lsn.receptive_field)
        elif trigger == 6:
            print(lsn.extralength)
            print(lsn.pval)
            print(lsn.receptive_field)

        assert lsn._stim_table is not StimulusAnalysis._PRELOAD
        assert lsn._LSN is not StimulusAnalysis._PRELOAD
        assert lsn._LSN_mask is not StimulusAnalysis._PRELOAD
        assert lsn._sweeplength is not StimulusAnalysis._PRELOAD
        assert lsn._interlength is not StimulusAnalysis._PRELOAD
        assert lsn._extralength is not StimulusAnalysis._PRELOAD
        assert lsn._sweep_response is not StimulusAnalysis._PRELOAD
        assert lsn._mean_sweep_response is not StimulusAnalysis._PRELOAD
        assert lsn._pval is not StimulusAnalysis._PRELOAD
        assert lsn._receptive_field is not StimulusAnalysis._PRELOAD

        # verify super class members weren't preloaded
        assert not dataset.get_corrected_fluorescence_traces.called
        assert lsn._timestamps is StimulusAnalysis._PRELOAD
        assert lsn._celltraces is StimulusAnalysis._PRELOAD
        assert lsn._numbercells is StimulusAnalysis._PRELOAD

        assert not dataset.get_roi_ids.called
        assert lsn._roi_id is StimulusAnalysis._PRELOAD

        assert not dataset.get_cell_specimen_ids.called
        assert lsn._cell_id is StimulusAnalysis._PRELOAD

        assert not dataset.get_dff_traces.called
        assert lsn._dfftraces is StimulusAnalysis._PRELOAD

        assert lsn._dxcm is StimulusAnalysis._PRELOAD
        assert lsn._dxtime is StimulusAnalysis._PRELOAD
