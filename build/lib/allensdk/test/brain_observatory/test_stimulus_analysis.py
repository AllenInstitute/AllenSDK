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


@pytest.mark.parametrize('trigger',
                         (1,2,3,4,5))
def test_harness(dataset,
                 trigger):
    with patch('allensdk.brain_observatory.stimulus_analysis.StimulusAnalysis.get_speed_tuning',
               mock_speed_tuning()) as get_speed_tuning:
        sa = StimulusAnalysis(dataset)

        assert sa._timestamps == StimulusAnalysis._PRELOAD
        assert sa._celltraces == StimulusAnalysis._PRELOAD
        assert sa._numbercells == StimulusAnalysis._PRELOAD
        assert sa._roi_id == StimulusAnalysis._PRELOAD
        assert sa._cell_id == StimulusAnalysis._PRELOAD
        assert sa._dfftraces == StimulusAnalysis._PRELOAD
        assert sa._dxcm == StimulusAnalysis._PRELOAD
        assert sa._dxtime == StimulusAnalysis._PRELOAD

        if trigger == 1:
            print(sa.timestamps)
            print(sa.dxcm)
            print(sa.binned_dx_sp)
        elif trigger == 2:
            print(sa.celltraces)
            print(sa.dxtime)
            print(sa.binned_cells_sp)
        elif trigger == 3:
            print(sa.acquisition_rate)
            print(sa.dxcm)
            print(sa.binned_dx_vis)
        elif trigger == 4:
            print(sa.numbercells)
            print(sa.dxtime)
            print(sa.binned_cells_vis)
        elif trigger == 5:
            print(sa.timestamps)
            print(sa.dxcm)
            print(sa.peak_run)

        print(sa.roi_id)
        print(sa.cell_id)
        print(sa.dfftraces)

        dataset.get_corrected_fluorescence_traces.assert_called_once_with()
        assert sa._timestamps is not StimulusAnalysis._PRELOAD
        assert sa._celltraces is not StimulusAnalysis._PRELOAD
        assert sa._numbercells is not StimulusAnalysis._PRELOAD

        dataset.get_roi_ids.assert_called_once_with()
        assert sa._roi_id is not StimulusAnalysis._PRELOAD

        dataset.get_cell_specimen_ids.assert_called_once_with()
        assert sa._cell_id is not StimulusAnalysis._PRELOAD

        dataset.get_dff_traces.assert_called_once_with()
        assert sa._dfftraces is not StimulusAnalysis._PRELOAD

        assert sa._dxcm is not StimulusAnalysis._PRELOAD
        assert sa._dxtime is not StimulusAnalysis._PRELOAD

        get_speed_tuning.assert_called_once_with(binsize=800)
        assert sa._binned_dx_sp is not StimulusAnalysis._PRELOAD
        assert sa._binned_cells_sp is not StimulusAnalysis._PRELOAD
        assert sa._binned_dx_vis is not StimulusAnalysis._PRELOAD
        assert sa._binned_cells_vis is not StimulusAnalysis._PRELOAD
        assert sa._peak_run is not StimulusAnalysis._PRELOAD
