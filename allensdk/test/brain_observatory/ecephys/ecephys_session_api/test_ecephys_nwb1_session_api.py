from pathlib import Path

import pytest
import pandas as pd
import xarray as xr

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("nwb_path", [
    Path("/", "allen", "aibs", "mat", "Kael", "ecephys_data", "mouse412792.spikes.nwb")
])
def test_spikes_nwb1(nwb_path):
    """
    This test was based on the file /allen/aibs/mat/ecephys_data/mouse412792.spikes.nwb. To run this test please copy
    or create a link to it in this directory.
    """

    if not nwb_path.exists():
        pytest.skip()

    # TODO: Convert this NWB 1 file into a NWB 2 file that way we can check that NWB Adaptors return the same data
    #  and computations (minus a few exceptions for missing NWB 1 data).
    session = EcephysSession.from_nwb_path(path=str(nwb_path), nwb_version=1)
    assert(isinstance(session.units, pd.DataFrame))
    assert(len(session.units) == 1363)

    print(session.stimulus_names)

    assert(isinstance(session.stimulus_presentations, pd.DataFrame))
    assert(len(session.stimulus_presentations) == 70390)
    assert(len(session.get_stimulus_table(['Natural Images_5'])) == 5950)
    assert(len(session.get_stimulus_table(['drifting_gratings_2'])) == 630)
    assert(len(session.get_stimulus_table(['flash_250ms_1'])) == 150)
    assert(len(session.get_stimulus_table(['gabor_20_deg_250ms_0'])) == 3645)
    assert(len(session.get_stimulus_table(['natural_movie_one_three'])) == 18000)
    assert(len(session.get_stimulus_table(['natural_movie_three_four'])) == 36000)
    assert(len(session.get_stimulus_table(['spontaneous'])) == 15)
    assert(len(session.get_stimulus_table(['static_gratings_6'])) == 6000)

    assert(session.running_speed.shape[0] == 365700)

    assert(len(session.spike_times.keys()) == 1363)

    assert(len(session.mean_waveforms.keys()) == 1363)
    one_waveform = next(iter(session.mean_waveforms.values()))
    assert(isinstance(one_waveform, xr.DataArray))

    assert(len(session.probes) == 6)
    assert(len(session.channels) == 737)

    pst = session.presentationwise_spike_times()
    assert(isinstance(pst, pd.DataFrame) and len(pst) > 0)

    cpc = session.conditionwise_spike_statistics(
        stimulus_presentation_ids=session.stimulus_presentations.index.values[:40]
    )
    assert(isinstance(cpc, pd.DataFrame) and len(cpc) > 0)


