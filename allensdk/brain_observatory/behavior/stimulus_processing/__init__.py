import pickle
import warnings
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.stimulus_processing \
    .stimulus_templates import \
    StimulusTemplate, StimulusTemplateFactory
from allensdk.brain_observatory.behavior.stimulus_processing.util import \
    convert_filepath_caseinsensitive, get_image_set_name


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
    # workaround to rename columns to harmonize with visual
    # coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number',
                          value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(
        columns={'frame': 'start_frame',
                 'time': 'start_time',
                 'flash_number': 'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[int(start_frame)]
                                 for start_frame in
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
        These correspond to paths to image arrays presented, metadata
        on the whole set of images, and metadata on specific images,
        respectively.

    """
    # Sometimes the source is a zipped pickle:
    pkl_stimuli = pkl["items"]["behavior"]["stimuli"]
    metadata = {'image_set': pkl_stimuli["images"]["image_path"]}

    # Get image file name;
    # These are encoded case-insensitive in the pickle file :/
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
                orientation=np.NaN,
                phase=np.NaN,
                spatial_frequency=np.NaN,
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


def get_gratings_metadata(stimuli: Dict, start_idx: int = 0) -> pd.DataFrame:
    """
    This function returns the metadata for each unique grating that was
    presented during the experiment. If no gratings were displayed during
    this experiment it returns an empty dataframe with the expected columns.
    Parameters
    ----------
    stimuli:
        The stimuli field (pkl['items']['behavior']['stimuli']) loaded
        from the experiment pkl file.
    start_idx:
        The index to start index column

    Returns
    -------
    pd.DataFrame:
        DataFrame containing the unique stimuli presented during an
        experiment. The columns contained in this DataFrame are
        'image_category', 'image_name', 'image_set', 'phase',
        'spatial_frequency', 'orientation', and 'image_index'.
        This returns empty if no gratings were presented.

    """
    if 'grating' in stimuli:
        phase = stimuli['grating']['phase']
        correct_freq = stimuli['grating']['sf']
        set_logs = stimuli['grating']['set_log']
        unique_oris = set([set_log[1] for set_log in set_logs])

        image_names = []

        for unique_ori in unique_oris:
            image_names.append(f"gratings_{float(unique_ori)}")

        grating_dict = {
            'image_category': ['grating'] * len(unique_oris),
            'image_name': image_names,
            'orientation': list(unique_oris),
            'image_set': ['grating'] * len(unique_oris),
            'phase': [phase] * len(unique_oris),
            'spatial_frequency': [correct_freq] * len(unique_oris),
            'image_index': range(start_idx, start_idx + len(unique_oris), 1)
        }
        grating_df = pd.DataFrame.from_dict(grating_dict)
    else:
        grating_df = pd.DataFrame(columns=['image_category',
                                           'image_name',
                                           'image_set',
                                           'phase',
                                           'spatial_frequency',
                                           'orientation',
                                           'image_index'])
    return grating_df


def get_stimulus_templates(pkl: dict,
                           grating_images_dict: Optional[dict] = None
                           ) -> Optional[StimulusTemplate]:
    """
    Gets images presented during experiments from the behavior stimulus file
    (*.pkl)

    Parameters
    ----------
    pkl : dict
        Loaded pkl dict containing data for the presented stimuli.
    grating_images_dict : Optional[dict]
        Because behavior pkl files do not contain image versions of grating
        stimuli, they must be obtained from an external source. The
        grating_images_dict is a nested dictionary where top level keys
        correspond to grating image names (e.g. 'gratings_0.0',
        'gratings_270.0') as they would appear in table returned by
        get_gratings_metadata(). Sub-nested dicts are expected to have 'warped'
        and 'unwarped' keys where values are numpy image arrays
        of aforementioned warped or unwarped grating stimuli.

    Returns
    -------
    StimulusTemplate:
        StimulusTemplate object containing images that were presented during
        the experiment

    """

    pkl_stimuli = pkl['items']['behavior']['stimuli']
    if 'images' in pkl_stimuli:
        images = get_images_dict(pkl)
        image_set_filepath = images['metadata']['image_set']
        image_set_name = get_image_set_name(image_set_path=image_set_filepath)
        image_set_name = convert_filepath_caseinsensitive(
            image_set_name)

        return StimulusTemplateFactory.from_unprocessed(
            image_set_name=image_set_name,
            image_attributes=images['image_attributes'],
            images=images['images']
        )
    elif 'grating' in pkl_stimuli:
        if (grating_images_dict is None) or (not grating_images_dict):
            raise RuntimeError("The 'grating_images_dict' param MUST "
                               "be provided to get stimulus templates "
                               "because this pkl data contains "
                               "gratings presentations.")
        gratings_metadata = get_gratings_metadata(
            pkl_stimuli).to_dict(orient='records')

        unwarped_images = []
        warped_images = []
        for image_attrs in gratings_metadata:
            image_name = image_attrs['image_name']
            grating_imgs_sub_dict = grating_images_dict[image_name]
            unwarped_images.append(grating_imgs_sub_dict['unwarped'])
            warped_images.append(grating_imgs_sub_dict['warped'])

        return StimulusTemplateFactory.from_processed(
            image_set_name='grating',
            image_attributes=gratings_metadata,
            unwarped=unwarped_images,
            warped=warped_images
        )
    else:
        warnings.warn(
            "Could not determine stimulus template images from pkl file. "
            f"The pkl stimuli nested dict "
            "(pkl['items']['behavior']['stimuli']) contained neither "
            "'images' nor 'grating' but instead: "
            f"'{pkl_stimuli.keys()}'"
        )
        return None


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
        image_category, image_name, image_set, phase, spatial_frequency,
        orientation, and image index.

    """
    stimuli = pkl['items']['behavior']['stimuli']
    if 'images' in stimuli:
        images = get_images_dict(pkl)
        stimulus_index_df = pd.DataFrame(images['image_attributes'])
        image_set_filename = convert_filepath_caseinsensitive(
            images['metadata']['image_set'])
        stimulus_index_df['image_set'] = get_image_set_name(
            image_set_path=image_set_filename)
    else:
        stimulus_index_df = pd.DataFrame(columns=[
            'image_name', 'image_category', 'image_set', 'phase',
            'spatial_frequency', 'image_index'])

    # get the grating metadata will be empty if gratings are absent
    grating_df = get_gratings_metadata(stimuli,
                                       start_idx=len(stimulus_index_df))
    stimulus_index_df = stimulus_index_df.append(grating_df,
                                                 ignore_index=True,
                                                 sort=False)

    # Add an entry for omitted stimuli
    omitted_df = pd.DataFrame({'image_category': ['omitted'],
                               'image_name': ['omitted'],
                               'image_set': ['omitted'],
                               'orientation': np.NaN,
                               'phase': np.NaN,
                               'spatial_frequency': np.NaN,
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
                # following frame, so we need to bump the
                # epoch_start & epoch_end to get the timing right
                epoch_start += 1
                epoch_end += 1

                visual_stimuli_data.append({
                    "orientation": orientation,
                    "image_name": image_name,
                    "frame": epoch_start,
                    "end_frame": epoch_end,
                    "time": time[epoch_start],
                    "duration": time[epoch_end] - time[epoch_start],
                    # this will always work because an epoch
                    # will never occur near the end of time
                    "omitted": False,
                })

    visual_stimuli_df = pd.DataFrame(data=visual_stimuli_data)

    # Add omitted flash info:
    try:
        omitted_flash_frame_log = \
            data['items']['behavior']['omitted_flash_frame_log']
    except KeyError:
        # For sessions for which there were no omitted flashes
        omitted_flash_frame_log = dict()

    omitted_flash_list = []
    for _, omitted_flash_frames in omitted_flash_frame_log.items():
        stim_frames = visual_stimuli_df['frame'].values
        omitted_flash_frames = np.array(omitted_flash_frames)

        # Test offsets of omitted flash frames
        # to see if they are in the stim log
        offsets = np.arange(-3, 4)
        offset_arr = np.add(
            np.repeat(omitted_flash_frames[:, np.newaxis],
                      offsets.shape[0], axis=1),
            offsets)
        matched_any_offset = np.any(np.isin(offset_arr, stim_frames), axis=1)

        #  Remove omitted flashes that also exist in the stimulus log
        was_true_omitted = np.logical_not(matched_any_offset)  # bool
        omitted_flash_frames_to_keep = omitted_flash_frames[was_true_omitted]

        # Have to remove frames that are double-counted in omitted log
        omitted_flash_list += list(np.unique(omitted_flash_frames_to_keep))

    omitted = np.ones_like(omitted_flash_list).astype(bool)
    time = [time[fi] for fi in omitted_flash_list]
    omitted_df = pd.DataFrame({'omitted': omitted,
                               'frame': omitted_flash_list,
                               'time': time,
                               'image_name': 'omitted'})

    df = pd.concat((visual_stimuli_df, omitted_df),
                   sort=False).sort_values('frame').reset_index()
    return df


def is_change_event(stimulus_presentations: pd.DataFrame) -> pd.Series:
    """
    Returns whether a stimulus is a change stimulus
    A change stimulus is defined as the first presentation of a new image_name
    Omitted stimuli are ignored
    The first stimulus in the session is ignored

    :param stimulus_presentations
        The stimulus presentations table

    :return: is_change: pd.Series indicating whether a given stimulus is a
        change stimulus
    """
    stimuli = stimulus_presentations['image_name']

    # exclude omitted stimuli
    stimuli = stimuli[~stimulus_presentations['omitted']]

    prev_stimuli = stimuli.shift()

    # exclude first stimulus
    stimuli = stimuli.iloc[1:]
    prev_stimuli = prev_stimuli.iloc[1:]

    is_change = stimuli != prev_stimuli

    # reset back to original index
    is_change = is_change \
        .reindex(stimulus_presentations.index) \
        .rename('is_change')

    # Excluded stimuli are not change events
    is_change = is_change.fillna(False)

    return is_change
