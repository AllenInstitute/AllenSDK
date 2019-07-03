import pytest
import numpy as np
import pandas as pd
from mock import MagicMock

from allensdk.brain_observatory.ecephys import natural_scenes as ns
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
    images = np.empty(119*3*2)
    images[0:(119*3)] = np.repeat(np.arange(-1, 118), 3)
    images[(119*3):] = np.nan
    return pd.DataFrame({'Image': images,
                         'stimulus_name': ['Natural images_5']*119*3 + ['spontaneous']*119*3,
                         'start_time': np.linspace(5000.0, 5060.0, 119*3*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 119*3*2) + 0.25,
                         'duration': 0.25})


def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    ns_obj = ns.NaturalScenes(ecephys_session)
    assert(isinstance(ns_obj.stim_table, pd.DataFrame))
    assert(len(ns_obj.stim_table) == 119*3)
    assert(ns_obj.number_images == 119)
    assert(ns_obj.number_nonblank == 118)
    assert(ns_obj.numbercells == 20)
    assert(ns_obj.mean_sweep_events.shape == (119*3, 20))



@pytest.mark.parametrize('responses,number_nonblank,expected',
                         [
                             (np.array([1.04, 2.8, 0.88, 2.08, 2.0, 0.48, 1.36, 2.72, 1.04, 0.96, 2.16, 0.56, 0.56, 0.96, 1.28, 1.52, 1.28, 1.36, 2.56, 1.76, 1.36, 1.04, 2.16, 1.2, 0.96, 0.88, 1.28, 2.88,
                                        1.28, 2.0, 1.6, 2.32, 1.84, 0.8, 1.04, 2.08, 2.56, 1.28, 1.04, 0.72, 1.6, 1.6, 1.44, 0.8, 1.52, 1.28, 2.4, 1.52, 1.04, 2.72, 0.96, 1.44, 1.04, 1.6, 2.24, 1.28, 1.2,
                                        1.44, 0.64, 1.36, 1.68, 0.72, 5.36, 1.12, 1.84, 0.88, 2.8, 1.84, 1.36, 1.04, 3.44, 1.28, 1.36, 2.16, 1.28, 1.36, 1.52, 0.88, 0.88, 1.12, 1.36, 0.8, 2.48, 1.28, 1.6,
                                        1.52, 1.04, 0.48, 0.8, 0.48, 2.48, 2.48, 1.2, 0.72, 1.28, 2.16, 1.68, 0.8, 1.28, 3.68, 1.6, 1.2, 1.92, 1.28, 1.6, 1.28, 3.12, 3.28, 2.0, 0.88, 1.92, 2.32, 0.88, 1.6,
                                        1.84, 1.12, 1.68, 1.36, 1.2]),
                              118, 0.559033898),
                             (np.zeros(119), 118, 1.0)
                         ])
def test_get_image_selectivity(responses, number_nonblank, expected):
    assert(np.isclose(ns.get_image_selectivity(responses, number_nonblank), expected))
