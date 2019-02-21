import numpy as np
import pandas as pd

from visual_behavior.translator import foraging2, foraging


def get_stimtable(core_data, timestamps_stimulus):

    stimulus_table = core_data['visual_stimuli'][:-10]  # ignore last 10 flashes
    # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(columns={'frame': 'start_frame', 'time': 'start_time'})
    stimulus_table.start_time = [timestamps_stimulus[start_frame] for start_frame in stimulus_table.start_frame.values]
    end_time = [timestamps_stimulus[end_frame] for end_frame in stimulus_table.end_frame.values]
    stimulus_table.insert(loc=4, column='end_time', value=end_time)

    return stimulus_table
    

def get_images_dict(pkl):

    try:
        images = foraging2.data_to_images(pkl)
    except KeyError:
        images = foraging.load_images(pkl)

    return images


def get_stimulus_template(pkl):

    images = get_images_dict(pkl)
    return np.array(images['images'])


def get_stimulus_metadata(pkl):

    images = get_images_dict(pkl)
    return pd.DataFrame(images['image_attributes'])