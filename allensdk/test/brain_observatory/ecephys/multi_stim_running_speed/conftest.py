import pytest
import numpy as np


@pytest.fixture
def basic_running_stim_file_fixture():
    rng = np.random.default_rng()
    return {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": rng.random((100,)),
                        "vsig": rng.uniform(low=0.0, high=5.1, size=(100,)),
                        "vin": rng.uniform(low=4.9, high=5.0, size=(100,)),
                    }]}}}
