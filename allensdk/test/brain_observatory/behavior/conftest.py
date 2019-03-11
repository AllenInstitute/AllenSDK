import sys
import pandas as pd
import pytest
import numpy as np


def pytest_ignore_collect(path, config):
    ''' The brain_observatory.ecephys submodule uses python 3.6 features that may not be backwards compatible!
    '''

    if sys.version_info < (3, 6):
        return True
    return False


@pytest.fixture
def running_data_df(running_speed):

    v_sig = np.ones_like(running_speed.values)
    v_in = np.ones_like(running_speed.values)
    dx = np.ones_like(running_speed.values)

    return pd.DataFrame({'v_sig': v_sig,
                         'v_in': v_in,
                         'speed': running_speed.values,
                         'dx': dx}, index=pd.Index(running_speed.timestamps, name='timestamps'))
