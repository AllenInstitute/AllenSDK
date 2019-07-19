import pytest
import numpy as np
import pandas as pd
import itertools
from mock import MagicMock

from allensdk.brain_observatory.ecephys import drifting_gratings as dg
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
    orivals = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    tfvals = [1.0, 2.0, 4.0, 8.0, 15.0]
    fmatrix = np.zeros((40*2, 2))
    fmatrix[0:40, :] = np.array(list(itertools.product(orivals, tfvals)))
    return pd.DataFrame({'Ori': fmatrix[:, 0], 'TF': fmatrix[:, 1],
                         'stimulus_name': ['drifting_gratings']*40 + ['spontaneous']*40,
                         'start_time': np.linspace(5000.0, 5060.0, 40*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 40*2) + 0.25,
                         'duration': 0.25})


def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    dg_obj = dg.DriftingGratings(ecephys_session)
    assert(isinstance(dg_obj.stim_table, pd.DataFrame))
    assert(len(dg_obj.stim_table) == 40)
    assert(dg_obj.number_tf == 5)
    assert(np.all(dg_obj.tfvals == [1.0, 2.0, 4.0, 8.0, 15.0]))
    assert(dg_obj.number_ori == 8)
    assert (np.all(dg_obj.orivals == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]))
    assert(dg_obj.numbercells == 20)
    assert(dg_obj.mean_sweep_events.shape == (40, 20))


@pytest.mark.parametrize('response,trials,bias,expected',
                         [
                             (np.array([1.233, 2.4, 0.5667, 0.8, 0.5]), np.array([1.5, 2.5, 1.0, 0.5, 0.0, 0.5, 2.0, 4.5, 2.5, 0.5, 1.5, 1.0, 0.5, 1.5, 0.0, 0.5]), 5, 0.4104327122),
                             (np.array([0.1, 0.233, 0.0333, 0.0667, 0.0333]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5, 1.0)
                         ])
def test_get_tfdi(response, trials, bias, expected):
    assert(np.isclose(dg.get_tfdi(response, trials, bias), expected))


@pytest.mark.parametrize('peak_response,all_responses,blank_responses,expected',
                         [
                             (0.2334, np.array([[0.0, 0.0, 0.0333, 0.0333, 0.0333],
                                                [0.033, 0.1, 0.033, 0.0333, 0.0333],
                                                [0.067, 0.0, 0.067, 0.0667, 0.0333],
                                                [0.033, 0.1, 0.067, 0.0, 0.0],
                                                [0.167, 0.033, 0.1, 0.133, 0.033],
                                                [0.0, 0.033, 0.0, 0.0, 0.1],
                                                [0.1, 0.233, 0.033, 0.0667, 0.033],
                                                [0.0, 0.0667, 0.067, 0.0667, 0.0]]), 0.03333333333333333, (0.20006666666666667, 0.01745666666666667) )
                         ])
def test_get_suppressed_contrast(peak_response, all_responses, blank_responses, expected):
    assert(np.allclose(dg.get_suppressed_contrast(peak_response, all_responses, blank_responses), expected))


@pytest.mark.parametrize('tuning_responses,tf_values,pref_tf_index,expected',
                         [
                             (np.array([0.46667, 0.16667, 1.7, 0.9333, 0.4]), [1.0, 2.0, 4.0, 8.0, 15.0], 2, (2.282457502110048, 4.865059677849396, 2.8284271247461903, np.nan)),
                             (np.array([17.633333, 17.96666667, 10.73333333, 12.63333333, 16.2]), [1.0, 2.0, 4.0, 8.0, 15.0], 1, (np.nan, np.nan, np.nan, np.nan)),
                             (np.array([3.26666667, 2.7, 2.53333333, 0.83333333, 1.3]), [1.0, 2.0, 4.0, 8.0, 15.0], 0, (0, 1.0, np.nan, 6.498019170849885))
                         ])
def test_fit_tf_tuning(tuning_responses, tf_values, pref_tf_index, expected):
    assert(np.allclose(dg.get_fit_tf_tuning(tuning_responses, tf_values, pref_tf_index), expected, equal_nan=True))
