import pandas as pd

import numpy as np

def get_licks(data, time):

    vsyncs = data["items"]["behavior"]['intervalsms']
    time = np.hstack((0, vsyncs)).cumsum() / 1000.0

    lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
    lick_times = [time[frame] for frame in lick_frames]
    return pd.DataFrame(data={"frame": lick_frames, "time": lick_times, })