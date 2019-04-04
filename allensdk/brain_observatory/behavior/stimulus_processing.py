import numpy as np
import pandas as pd
import pickle
from allensdk.brain_observatory.behavior import IMAGE_SETS

IMAGE_SETS_REV = {val:key for key, val in IMAGE_SETS.items()}


def load_pickle(pstream):
    return pickle.load(pstream, encoding="bytes")


def get_stimulus_presentations(data, stimulus_timestamps):

    visual_stimuli = get_visual_stimuli_df(data, stimulus_timestamps)
    stimulus_table = visual_stimuli[:-10]  # ignore last 10 flashes
    # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(columns={'frame': 'start_frame', 'time': 'start_time', 'flash_number':'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[start_frame] for start_frame in stimulus_table.start_frame.values]
    end_time = [stimulus_timestamps[end_frame] for end_frame in stimulus_table.end_frame.values]
    stimulus_table.insert(loc=4, column='stop_time', value=end_time)
    stimulus_table.set_index('stimulus_presentations_id', inplace=True)
    stimulus_table = stimulus_table[sorted(stimulus_table.columns)]

    return stimulus_table


def get_images_dict(pkl):

    # Sometimes the source is a zipped pickle:
    metadata = {'image_set': pkl["items"]["behavior"]["stimuli"]["images"]["image_path"]}

    image_set = load_pickle(open(metadata['image_set'], 'rb'))
    images = []
    images_meta = []

    ii = 0
    for cat, cat_images in image_set.items():
        for img_name, img in cat_images.items():
            meta = dict(
                image_category=cat.decode("utf-8"),
                image_name=img_name.decode("utf-8"),
                image_index=ii,
            )

            images.append(img)
            images_meta.append(meta)

            ii += 1

    images_dict = dict(
        metadata=metadata,
        images=images,
        image_attributes=images_meta,
    )

    return images_dict


def get_stimulus_templates(pkl):

    images = get_images_dict(pkl)
    image_set_filename = images['metadata']['image_set']
    return {IMAGE_SETS_REV[image_set_filename]: np.array(images['images'])}


def get_stimulus_metadata(pkl):

    images = get_images_dict(pkl)
    stimulus_index_df = pd.DataFrame(images['image_attributes'])
    stimulus_index_df['image_set'] = IMAGE_SETS_REV[images['metadata']['image_set']]
    stimulus_index_df.set_index(['image_index'], inplace=True, drop=True)
    return stimulus_index_df


def _resolve_image_category(change_log, frame):

    for change in (unpack_change_log(c) for c in change_log):
        if frame < change['frame']:
            return change['from_category']

    return change['to_category']

def _get_stimulus_epoch(set_log, current_set_index, start_frame, n_frames):
    try:
        next_set_event = set_log[current_set_index + 1]  # attr_name, attr_value, time, frame
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames, )

    return (start_frame, next_set_event[3])  # end frame isnt inclusive

def _get_draw_epochs(draw_log, start_frame, stop_frame):
    """start_frame inclusive, stop_frame non-inclusive
    """
    draw_epochs = []
    current_frame = start_frame

    while current_frame <= stop_frame:
        epoch_length = 0
        while current_frame < stop_frame and draw_log[current_frame] == 1:
            epoch_length += 1
            current_frame += 1
        else:
            current_frame += 1

        if epoch_length:
            draw_epochs.append(
                (current_frame - epoch_length - 1, current_frame - 1, )
            )

    return draw_epochs

def unpack_change_log(change):
    
    (from_category, from_name), (to_category, to_name, ), time, frame = change

    return dict(
        frame=frame,
        time=time,
        from_category=from_category,
        to_category=to_category,
        from_name=from_name,
        to_name=to_name,
    )

def get_visual_stimuli_df(data, time):

    stimuli = data['items']['behavior']['stimuli']
    n_frames = len(time)
    visual_stimuli_data = []
    for stimuli_group_name, stim_dict in stimuli.items():
        for idx, (attr_name, attr_value, _time, frame, ) in \
                enumerate(stim_dict["set_log"]):
            orientation = attr_value if attr_name.lower() == "ori" else np.nan
            image_name = attr_value if attr_name.lower() == "image" else np.nan

            if attr_name.lower() == "image" and stim_dict["change_log"]:
                image_category = _resolve_image_category(
                    stim_dict["change_log"],
                    frame
                )
            else:
                image_category = np.nan

            stimulus_epoch = _get_stimulus_epoch(
                stim_dict["set_log"],
                idx,
                frame,
                n_frames,
            )
            draw_epochs = _get_draw_epochs(
                stim_dict["draw_log"],
                *stimulus_epoch
            )

            for idx, (epoch_start, epoch_end, ) in enumerate(draw_epochs):

                # visual stimulus doesn't actually change until start of
                # following frame, so we need to bump the epoch_start & epoch_end
                # to get the timing right
                epoch_start += 1
                epoch_end += 1

                visual_stimuli_data.append({
                    "orientation": orientation,
                    "image_name": image_name,
                    "image_category": image_category,
                    "frame": epoch_start,
                    "end_frame": epoch_end,
                    "time": time[epoch_start],
                    "duration": time[epoch_end] - time[epoch_start],  # this will always work because an epoch will never occur near the end of time
                })

    return pd.DataFrame(data=visual_stimuli_data)