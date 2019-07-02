import pytest
import numpy as np
import pandas as pd
import itertools
from mock import MagicMock, patch

from allensdk.brain_observatory.ecephys import static_gratings as sg
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


@pytest.fixture
def ecephys_session():
    ecephys_ses = MagicMock(spec=EcephysSession)
    units_df = pd.DataFrame({'unit_id': np.arange(20)})
    units_df = units_df.set_index('unit_id')
    ecephys_ses.units = units_df
    ecephys_ses.spike_times = {uid: np.linspace(0, 1.0, 5) for uid in np.arange(20)}
    return ecephys_ses

@pytest.fixture
def stimulus_table():
    orival = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    sfvals = [0.02, 0.04, 0.08, 0.16, 0.32]
    phasevals = [0.0, 0.25, 0.50, 0.75]
    fmatrix = np.zeros((120*2, 3))
    fmatrix[0:120, :] = np.array(list(itertools.product(orival, sfvals, phasevals)))
    return pd.DataFrame({'Ori': fmatrix[:, 0], 'SF': fmatrix[:, 1], 'Phase': fmatrix[:, 2],
                         'stimulus_name': ['static_gratings_6']*120 + ['spontaneous']*120,
                         'start_time': np.linspace(5000.0, 5060.0, 120*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 120*2) + 0.25,
                         'duration': 0.25})

# @patch.object(EcephysSession, 'stimulus_presentations', stimulus_table())
def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    sg_obj = sg.StaticGratings(ecephys_session)
    assert(isinstance(sg_obj.stim_table, pd.DataFrame))
    assert(len(sg_obj.stim_table) == 120)
    assert(sg_obj.number_sf == 5)
    assert(np.all(sg_obj.sfvals == [0.02, 0.04, 0.08, 0.16, 0.32]))
    assert(sg_obj.number_ori == 6)
    assert (np.all(sg_obj.orivals == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]))
    assert(sg_obj.number_phase == 4)
    assert (np.all(sg_obj.phasevals == [0.0, 0.25, 0.50, 0.75]))
    assert(sg_obj.numbercells == 20)
    assert(sg_obj.mean_sweep_events.shape == (120, 20))


@pytest.mark.parametrize('sf_tuning_response,sf_vals,pref_sf_index,expected',
                         [
                             (np.array([2.69565217, 3.91836735, 2.36734694, 1.52, 2.21276596]), [0.02, 0.04, 0.08, 0.16, 0.32], 1, (0.22704947240176027, 0.0234087755414, np.nan, np.nan)),
                             (np.array([1.14285714, 0.73469388, 7.44, 13.6, 11.6]), [0.02, 0.04, 0.08, 0.16, 0.32], 3, (3.290141840632274, 0.1956416782774323, 0.08, np.nan)),
                             (np.array([2.24, 1.83333333, 1.68, 1.87755102, 1.87755102]), [0.02, 0.04, 0.08, 0.16, 0.32], 0, (0.0, 0.019999999552965164, np.nan, 0.32))
                         ])
def test_fit_sf_tuning(sf_tuning_response, sf_vals, pref_sf_index, expected):
    assert(np.allclose(sg.fit_sf_tuning(sf_tuning_response, sf_vals, pref_sf_index), expected, equal_nan=True))


@pytest.mark.parametrize('sf_tuning_responses,mean_sweeps_trials,expected',
                         [
                             (np.array([18.08333, 19.8333, 28.333, 14.80, 9.6170]),
                              np.array([12.0, 4.0, 8.0, 32.0, 4.0, 0.0, 4.0, 8.0, 24.0, 40.0, 32.0, 8.0, 20.0, 28.0, 24.0, 28.0, 0.0, 4.0, 4.0, 24.0, 16.0, 8.0, 16.0, 4.0, 0.0, 4.0,
                                        24.0, 4.0, 12.0, 20.0, 0.0, 12.0, 0.0, 16.0]), 0.4402349784724991)
                         ])
def test_get_sfdi(sf_tuning_responses, mean_sweeps_trials, expected):
    assert(sg.get_sfdi(sf_tuning_responses, mean_sweeps_trials, len(sf_tuning_responses)) == expected)


if __name__ == '__main__':
    test_fit_sf_tuning(np.array([2.69565217, 3.91836735, 2.36734694, 1.52, 2.21276596]), [0.02, 0.04, 0.08, 0.16, 0.32],
                       1, (0.22704947240176027, 0.0234087755414, np.nan, 0.32))
    #test_fit_sf_tuning(np.array([1.14285714, 0.73469388, 7.44, 13.6, 11.6]), [0.02, 0.04, 0.08, 0.16, 0.32],
    #                   3, (3.290141840632274, 0.1956416782774323, 0.08, np.nan))
    #test_fit_sf_tuning(np.array([2.24, 1.83333333, 1.68, 1.87755102, 1.87755102]), [0.02, 0.04, 0.08, 0.16, 0.32],
    #                   0, (0.0, 0.019999999552965164, np.nan, 0.32))

    #test_get_sfdi(np.array([18.08333, 19.8333, 28.333, 14.80, 9.6170]),
    #                       np.array([12.0, 4.0, 8.0, 32.0, 4.0, 0.0, 4.0, 8.0, 24.0, 40.0, 32.0, 8.0, 20.0, 28.0, 24.0, 28.0,
    #                        0.0, 4.0, 4.0, 24.0, 16.0,
    #                        8.0, 16.0, 4.0, 0.0, 4.0, 24.0, 4.0, 12.0, 20.0, 0.0, 12.0, 0.0, 16.0]),
    #              0.4402349784724991)

    #feature_matrix = np.empty((120, 3), dtype=np.float64)
    pass