import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Union

from allensdk.brain_observatory.behavior import IMAGE_SETS

IMAGE_SETS_REV = {val: key for key, val in IMAGE_SETS.items()}


def convert_filepath_caseinsensitive(filename_in):
    if filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_TRAINING_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_G_2019.05.26.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_G_2019.05.26.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_H_2019.05.26.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_H_2019.05.26.pkl'
    else:
        raise NotImplementedError(filename_in)


def load_pickle(pstream):
    return pickle.load(pstream, encoding="bytes")


def get_stimulus_presentations(data, stimulus_timestamps) -> pd.DataFrame:
    """
    This function retrieves the stimulus presentation dataframe and
    renames the columns, adds a stop_time column, and set's index to
    stimulus_presentation_id before sorting and returning the dataframe.
    :param data: stimulus file associated with experiment id
    :param stimulus_timestamps: timestamps indicating when stimuli switched
                                during experiment
    :return: stimulus_table: dataframe containing the stimuli metadata as well
                             as what stimuli was presented
    """
    stimulus_table = get_visual_stimuli_df(data, stimulus_timestamps)
    # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(
        columns={'frame': 'start_frame', 'time': 'start_time', 'flash_number': 'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[int(start_frame)] for start_frame in
                                 stimulus_table.start_frame.values]
    end_time = []
    for end_frame in stimulus_table.end_frame.values:
        if not np.isnan(end_frame):
            end_time.append(stimulus_timestamps[int(end_frame)])
        else:
            end_time.append(float('nan'))

    stimulus_table.insert(loc=4, column='stop_time', value=end_time)
    stimulus_table.set_index('stimulus_presentations_id', inplace=True)
    stimulus_table = stimulus_table[sorted(stimulus_table.columns)]
    return stimulus_table


def get_images_dict(pkl) -> Dict:
    """
    Gets the dictionary of images that were presented during an experiment
    along with image set metadata and the image specific metadata. This
    function uses the path to the image pkl file to read the images and their
    metadata from the pkl file and return this dictionary.
    Parameters
    ----------
    pkl: The pkl file containing the data for the stimuli presented during
         experiment

    Returns
    -------
    Dict:
        A dictionary containing keys images, metadata, and image_attributes.
        These correspond to paths to images to images presented, metadata
        on the whole set of images, and metadata on specific images,
        respectively.

    """
    # Sometimes the source is a zipped pickle:
    metadata = {'image_set': pkl["items"]["behavior"]["stimuli"]["images"]["image_path"]}

    # Get image file name; these are encoded case-insensitive in the pickle file :/
    filename = convert_filepath_caseinsensitive(metadata['image_set'])

    image_set = load_pickle(open(filename, 'rb'))
    images = []
    images_meta = []

    ii = 0
    for cat, cat_images in image_set.items():
        for img_name, img in cat_images.items():
            meta = dict(
                image_category=cat.decode("utf-8"),
                image_name=img_name.decode("utf-8"),
                phase=np.NaN,
                correct_frequency=np.NaN,
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


def get_stimulus_templates(pkl) -> Dict:
    """
    Gets dictionary of images presented during experimentation
    Parameters
    ----------
    pkl: pkl file containing the data for the presented stimuli

    Returns
    -------
    Dict:
        Dictionary of images that were presented during the experiment

    """
    images = get_images_dict(pkl)
    image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
    return {IMAGE_SETS_REV[image_set_filename]: np.array(images['images'])}


def get_stimulus_metadata(pkl) -> pd.DataFrame:
    """
    Gets the stimulus metadata for each type of stimulus presented during
    the experiment. The metadata is return for gratings, images, and omitted
    stimuli.
    Parameters
    ----------
    pkl: the pkl file containing the information about what stimuli were
         presented during the experiment

    Returns
    -------
    pd.DataFrame:
        The dataframe containing a row for every stimulus that was presented
        during the experiment. The row contains the following data,
        image_category, image_name, image_set, phase, correct_frequency,
        and image index.

    """
    stimuli = pkl['items']['behavior']['stimuli']
    if 'images' in stimuli:
        images = get_images_dict(pkl)
        stimulus_index_df = pd.DataFrame(images['image_attributes'])
        image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
        stimulus_index_df['image_set'] = IMAGE_SETS_REV[image_set_filename]
    else:
        stimulus_index_df = pd.DataFrame(columns=[
            'image_name', 'image_category', 'image_set', 'phase',
            'correct_frequency', 'image_index'])

    # if grating are in the pkl add an entry for each grating possible
    if 'grating' in stimuli:
        phase = stimuli['grating']['phase']
        correct_freq = stimuli['grating']['correct_freq']
        start_idx = len(stimulus_index_df)
        grating_df = {'image_category': ['grating']*4,
                      'image_name': ['gratings_0.0', 'gratings_90.0',
                                     'gratings_180.0', 'gratings_270.0'],
                      'image_set': ['grating']*4,
                      'phase': [phase]*4,
                      'correct_frequency': [correct_freq]*4,
                      'image_index': [start_idx, start_idx+1, start_idx+2,
                                       start_idx+3]}
        grating_df = pd.DataFrame.from_dict(grating_df)

        stimulus_index_df = stimulus_index_df.append(grating_df,
                                                     ignore_index=True,
                                                     sort=False)

    # Add an entry for omitted stimuli
    omitted_df = pd.DataFrame({'image_category': ['omitted'],
                               'image_name': ['omitted'],
                               'image_set': ['omitted'],
                               'phase': np.NaN,
                               'correct_frequency': np.NaN,
                               'image_index': len(stimulus_index_df)})
    stimulus_index_df = stimulus_index_df.append(omitted_df, ignore_index=True,
                                                 sort=False)
    stimulus_index_df.set_index(['image_index'], inplace=True, drop=True)
    return stimulus_index_df


def _resolve_image_category(change_log, frame):
    for change in (unpack_change_log(c) for c in change_log):
        if frame < change['frame']:
            return change['from_category']

    return change['to_category']


def _get_stimulus_epoch(set_log: List[Tuple[str, Union[str, int], int, int]],
                        current_set_index: int, start_frame: int,
                        n_frames: int) -> Tuple[int, int]:
    """
    Gets the frame range for which a stimuli was presented and the transition
    to the next stimuli was ongoing. Returns this in the form of a tuple.
    Parameters
    ----------
    set_log: List[Tuple[str, Union[str, int], int, int
        The List of Tuples in the form of
        (stimuli_type ('Image' or 'Grating'),
         stimuli_descriptor (image_name or orientation of grating in degrees),
         nonsynced_time_of_display (not sure, it's never used),
         display_frame (frame that stimuli was displayed))
    current_set_index: int
        Index of stimuli set to calculate window
    start_frame: int
        frame where stimuli was set, set_log[current_set_index][3]
    n_frames: int
        number of frames for which stimuli were displayed

    Returns
    -------
    Tuple[int, int]:
        A tuple where index 0 is start frame of stimulus window and index 1 is
        end frame of stimulus window

    """
    try:
        next_set_event = set_log[current_set_index + 1]
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames,)

    return start_frame, next_set_event[3]  # end frame isn't inclusive


def _get_draw_epochs(draw_log: List[int], start_frame: int,
                     stop_frame: int) -> List[Tuple[int, int]]:
    """
    Gets the frame numbers of the active frames within a stimulus window.
    Stimulus epochs come in the form [0, 0, 1, 1, 0, 0] where the stimulus is
    active for some amount of time in the window indicated by int 1 at that
    frame. This function returns the ranges for which the set_log is 1 within
    the draw_log window.
    Parameters
    ----------
    draw_log: List[int]
        A list of ints indicating for what frames stimuli were active
    start_frame: int
        The start frame to search within the draw_log for active values
    stop_frame: int
        The end frame to search within the draw_log for active values

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples indicating the start and end frames of every
        contiguous set of active values within the specified window
        of the draw log.
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
                (current_frame - epoch_length - 1, current_frame - 1,)
            )

    return draw_epochs


def unpack_change_log(change):
    (from_category, from_name), (to_category, to_name,), time, frame = change

    return dict(
        frame=frame,
        time=time,
        from_category=from_category,
        to_category=to_category,
        from_name=from_name,
        to_name=to_name,
    )


def get_visual_stimuli_df(data, time) -> pd.DataFrame:
    """
    This function loads the stimuli and the omitted stimuli into a dataframe.
    These stimuli are loaded from the input data, where the set_log and
    draw_log contained within are used to calculate the epochs. These epochs
    are used as start_frame and end_frame and converted to times by input
    stimulus timestamps. The omitted stimuli do not have a end_frame by design
    though there duration is always 250ms.
    :param data: the behavior data file
    :param time: the stimulus timestamps indicating when each stimuli is
                 displayed
    :return: df: a pandas dataframe containing the stimuli and omitted stimuli
                 that were displayed with their frame, end_frame, start_time,
                 and duration
    """

    stimuli = data['items']['behavior']['stimuli']
    n_frames = len(time)
    visual_stimuli_data = []
    for stimuli_group_name, stim_dict in stimuli.items():
        for idx, (attr_name, attr_value, _time, frame,) in \
                enumerate(stim_dict["set_log"]):
            orientation = attr_value if attr_name.lower() == "ori" else np.nan
            image_name = attr_value if attr_name.lower() == "image" else np.nan

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

            for idx, (epoch_start, epoch_end,) in enumerate(draw_epochs):
                # visual stimulus doesn't actually change until start of
                # following frame, so we need to bump the epoch_start & epoch_end
                # to get the timing right
                epoch_start += 1
                epoch_end += 1

                visual_stimuli_data.append({
                    "orientation": orientation,
                    "image_name": image_name,
                    "frame": epoch_start,
                    "end_frame": epoch_end,
                    "time": time[epoch_start],
                    "duration": time[epoch_end] - time[epoch_start],
                    # this will always work because an epoch will never occur near the end of time
                    "omitted": False,
                })

    visual_stimuli_df = pd.DataFrame(data=visual_stimuli_data)

    draw_log_rising_edges = 0
    draw_log_gratings = 0
    # if images are contained in stimuli
    if 'images' in stimuli.keys():
        # ensure that every rising edge in the draw_log is accounted for in the visual_stimuli_df
        draw_log_rising_edges = len(np.where(np.diff(stimuli['images']['draw_log']) == 1)[0])
    if 'grating' in stimuli.keys():
        draw_log_gratings = len(np.where(np.diff(stimuli['grating']['draw_log']) == 1)[0])

    discrete_flashes = len(visual_stimuli_data)
    assert (draw_log_rising_edges +
            draw_log_gratings) == discrete_flashes, "the number of rising edges in the draw log is expected to match the number of flashes in the stimulus table"

    # Add omitted flash info:
    omitted_flash_list = []
    omitted_flash_frame_log = data['items']['behavior']['omitted_flash_frame_log']

    for stimuli_group_name, omitted_flash_frames in omitted_flash_frame_log.items():
        stim_frames = visual_stimuli_df['frame'].values
        omitted_flash_frames = np.array(omitted_flash_frames)

        #  Test offsets of omitted flash frames to see if they are in the stim log
        offsets = np.arange(-3, 4)
        offset_arr = np.add(np.repeat(omitted_flash_frames[:, np.newaxis], offsets.shape[0], axis=1), offsets)
        matched_any_offset = np.any(np.isin(offset_arr, stim_frames), axis=1)

        #  Remove omitted flashes that also exist in the stimulus log
        was_true_omitted = np.logical_not(matched_any_offset)  # bool
        omitted_flash_frames_to_keep = omitted_flash_frames[was_true_omitted]

        # Have to remove frames that are double-counted in omitted log
        omitted_flash_list += list(np.unique(omitted_flash_frames_to_keep))

    omitted = np.ones_like(omitted_flash_list).astype(bool)
    time = [time[fi] for fi in omitted_flash_list]
    omitted_df = pd.DataFrame({'omitted': omitted, 'frame': omitted_flash_list, 'time': time,
                               'image_name': 'omitted'})

    df = pd.concat((visual_stimuli_df, omitted_df), sort=False).sort_values('frame').reset_index()
    return df
