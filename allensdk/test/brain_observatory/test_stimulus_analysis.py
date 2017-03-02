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
