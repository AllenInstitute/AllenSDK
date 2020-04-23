import pandas as pd

from allensdk.brain_observatory.behavior.rewards_processing import get_rewards


def test_get_rewards():
    data = {
        "items": {
            "behavior": {
                "trial_log": [
                    {
                        'rewards': [(0.007, 1085.96, 64775)],
                        'trial_params': {
                            'catch': False, 'auto_reward': False,
                            'change_time': 5}},
                    {
                        'rewards': [(0.007, 1090.01, 64780)],
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
        {"volume": [0.007, 0.007],
         "timestamps": [1086.96, 1091.01],
         "autorewarded": [False, True]}).set_index("timestamps", drop=True)

    pd.testing.assert_frame_equal(expected, get_rewards(data, lambda x: x+1.0))
