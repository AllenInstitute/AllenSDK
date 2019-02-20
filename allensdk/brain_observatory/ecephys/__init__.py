from typing import NamedTuple

import numpy as np


class RunningSpeed(NamedTuple):
    ''' Describes the rate at which an experimental subject ran during a session.

    values : np.ndarray
        running speed (cm/s) at each sample point
    timestamps : np.ndarray
        The time at which each sample was collected (s).

    '''

    timestamps : np.ndarray
    values: np.ndarray

