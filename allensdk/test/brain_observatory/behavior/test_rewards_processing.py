import pandas as pd
import numpy as np

from allensdk.brain_observatory.behavior.rewards_processing import get_rewards


def test_get_rewards():
    data = {
        "items": {
            "behavior": {
                "trial_log": [
                    {
                        'rewards': [(0.007, 1085.96, 55)],
                        'trial_params': {
                            'catch': False, 'auto_reward': False,
                            'change_time': 5}},
                    {
                        'rewards': [(0.008, 1090.01, 66)],
                        'trial_params': {
                            'catch': False, 'auto_reward': True,
                            'change_time': 6}},
                    {
                        'rewards': [],
                        'trial_params': {
                            'catch': False, 'auto_reward': False,
                            'change_time': 4},
                    },
                    ]
                }}}
    expected = pd.DataFrame(
        {"volume": [0.007, 0.008],
         "timestamps": [14.0, 15.0],
         "autorewarded": [False, True]})

    timesteps = -1*np.ones(100, dtype=float)
    timesteps[55] = 14.0
    timesteps[66] = 15.0
    pd.testing.assert_frame_equal(expected, get_rewards(data, timesteps))
