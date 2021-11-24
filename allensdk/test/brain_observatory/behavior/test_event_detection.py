import numpy as np
import pytest

from allensdk.brain_observatory.behavior.event_detection import \
    filter_events_array


def test_filter_events_array():
    with pytest.raises(ValueError):
        filter_events_array(arr=np.array([0.0, 0.0, 0.6]))

    with pytest.raises(ValueError):
        filter_events_array(arr=np.array([[0.0, 0.0, 0.6]]), n_time_steps=0)

    arr = np.array([[0.0, 0.0, 0.6]])
    filtered_events_array = filter_events_array(arr=arr)
    assert arr.shape[0] == filtered_events_array.shape[0]
    assert arr.shape[1] == filtered_events_array.shape[1]

    expected = np.array([[0.0, 0.0, 0.199559]])
    assert (np.abs(filtered_events_array - expected) < 1e-6).all()
