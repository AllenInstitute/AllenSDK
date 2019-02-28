import pandas as pd
from collections import defaultdict

def get_rewards(data, time, stimulus_rebase_function):
    trial_df = pd.DataFrame(data["items"]["behavior"]['trial_log'])
    rewards_dict = defaultdict(dict)
    for idx, trial in trial_df.iterrows():
        rewards = trial["rewards"]  # as i write this there can only ever be one reward per trial
        if rewards:
            rewards_dict["volume"][idx] = rewards[0][0]
            rewards_dict["time"][idx] = stimulus_rebase_function(time[rewards[0][2]])
            rewards_dict["lickspout"][idx] = None  # not yet implemented in the foraging2 output
    return pd.DataFrame(data=rewards_dict)