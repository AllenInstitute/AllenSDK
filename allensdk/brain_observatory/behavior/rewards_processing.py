import pandas as pd
from collections import defaultdict


def get_rewards(data, stimulus_rebase_function):
    trial_df = pd.DataFrame(data["items"]["behavior"]["trial_log"])
    rewards_dict = {"volume": [], "timestamps": [], "autorewarded": []}
    for idx, trial in trial_df.iterrows():
        rewards = trial["rewards"]  # as i write this there can only ever be one reward per trial
        if rewards:
            rewards_dict["volume"].append(rewards[0][0])
            rewards_dict["timestamps"].append(stimulus_rebase_function(rewards[0][1]))
            rewards_dict["autorewarded"].append(trial["trial_params"]["auto_reward"])

    df = pd.DataFrame(rewards_dict).set_index("timestamps", drop=True)

    return df
