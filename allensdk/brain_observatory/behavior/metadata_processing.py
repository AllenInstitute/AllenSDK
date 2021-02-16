OPHYS_0_HABITUATION = (
    "A behavior training session performed on the 2-photon calcium imaging "
    "setup but without recording neural activity, with the goal of "
    "habituating the mouse to the experimental setup before commencing "
    "imaging of neural activity. Habituation sessions are change detection "
    "with the same image set on which the mouse was trained. The session is "
    "75 minutes long, with 5 minutes of gray screen before and after 60 "
    "minutes of behavior, followed by 10 repeats of a 30 second natural "
    "movie stimulus at the end of the session."
)

OPHYS_1_3_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse brain as the "
    "mouse performs a visual change detection task with a set of natural "
    "images upon which it has been previously trained. Image stimuli are "
    "displayed for 250 ms with a 500 ms intervening gray period. 5% of "
    "non-change image presentations are randomly omitted. The session is "
    "75 minutes long, with 5 minutes of gray screen before and after 60 "
    "minutes of behavior, followed by 10 repeats of a 30 second natural "
    "movie stimulus at the end of the session."
)

OPHYS_2_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse brain as "
    "the mouse is passively shown changes in natural scene images upon which "
    "it was previously trained as the change detection task is played in "
    "open loop mode, with the lick-response sensory withdrawn and the mouse "
    "is unable to respond to changes or receive reward feedback. Image "
    "stimuli are displayed for 250 ms with a 500 ms intervening gray period. "
    "5% of non-change image presentations are randomly omitted. The session "
    "is 75 minutes long, with 5 minutes of gray screen before and after 60 "
    "minutes of behavior, followed by 10 repeats of a 30 second natural "
    "movie stimulus at the end of the session."
)

OPHYS_4_6_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse brain as the "
    "mouse performs a visual change detection task with natural scene images "
    "that are unique from those on which the mouse was trained prior to the "
    "imaging phase of the experiment. Image stimuli are displayed for 250 ms "
    "with a 500 ms intervening gray period. 5% of non-change image "
    "presentations are randomly omitted. The session is 75 minutes long, with "
    "5 minutes of gray screen before and after 60 minutes of behavior, "
    "followed by 10 repeats of a 30 second natural movie stimulus at the end "
    "of the session."
)

OPHYS_5_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse brain as the "
    "mouse is passively shown changes in natural scene images that are unique "
    "from those on which the mouse was trained prior to the imaging phase of "
    "the experiment. In this session, the change detection task is played in "
    "open loop mode, with the lick-response sensory withdrawn and the mouse "
    "is unable to respond to changes or receive reward feedback. Image "
    "stimuli are displayed for 250 ms with a 500 ms intervening gray period. "
    "5% of non-change image presentations are randomly omitted. The session "
    "is 75 minutes long, with 5 minutes of gray screen before and after 60 "
    "minutes of behavior, followed by 10 repeats of a 30 second natural movie "
    "stimulus at the end of the session."
)

TRAINING_GRATINGS_0 = (
    "An associative training session where a mouse is automatically rewarded "
    "when a grating stimulus changes orientation. Grating stimuli are "
    "full-field, square-wave static gratings with a spatial frequency of "
    "0.02 cycles per degree, with orientation changes between 0 and 90 "
    "degrees, at two spatial phases. Delivered rewards are 5ul in volume, "
    "and the session lasts for 15 minutes."
)

TRAINING_GRATINGS_1 = (
    "An operant behavior training session where a mouse must lick following "
    "a change in stimulus identity to earn rewards. Stimuli consist of "
    "full-field, square-wave static gratings with a spatial frequency of "
    "0.02 cycles per degree. Orientation changes between 0 and 90 degrees "
    "occur with no intervening gray period. Delivered rewards are 10ul in "
    "volume, and the session lasts 60 minutes"
)

TRAINING_GRATINGS_2 = (
    "An operant behavior training session where a mouse must lick following "
    "a change in stimulus identity to earn rewards. Stimuli consist of "
    "full-field, square-wave static gratings with a spatial frequency of "
    "0.02 cycles per degree. Gratings of 0 or 90 degrees are presented for "
    "250 ms with a 500 ms intervening gray period. Delivered rewards are "
    "10ul in volume, and the session lasts 60 minutes."
)

TRAINING_IMAGES_3 = (
    "An operant behavior training session where a mouse must lick following "
    "a change in stimulus identity to earn rewards. Stimuli consist of 8 "
    "natural scene images, for a total of 64 possible pairwise transitions. "
    "Images are shown for 250 ms with a 500 ms intervening gray period. "
    "Delivered rewards are 10ul in volume, and the session lasts for 60 "
    "minutes"
)

TRAINING_IMAGES_4 = (
    "An operant behavior training session where a mouse must lick a spout "
    "following a change in stimulus identity to earn rewards. Stimuli "
    "consist of 8 natural scene images, for a total of 64 possible pairwise "
    "transitions. Images are shown for 250 ms with a 500 ms intervening "
    "gray period. Delivered rewards are 7ul in volume, and the session "
    "lasts for 60 minutes"
)


def get_expt_description(session_type: str) -> str:
    """Determine a behavior ophys session's experiment description based on
    session type.

    Parameters
    ----------
    session_type : str
        A session description string (e.g. OPHYS_1_images_B )

    Returns
    -------
    str
        A description of the experiment based on the session_type.

    Raises
    ------
    RuntimeError
        Behavior ophys sessions should only have 6 different session types.
        Unknown session types (or malformed session_type strings) will raise
        an error.
    """
    # Experiment descriptions for different session types:
    # OPHYS_1 -> OPHYS_6
    description_dict = dict()
    description_dict.update({"OPHYS_0": OPHYS_0_HABITUATION})
    description_dict.update(
        dict.fromkeys(["OPHYS_1", "OPHYS_3"], OPHYS_1_3_DESCRIPTION))
    description_dict.update(
        dict.fromkeys(["OPHYS_4", "OPHYS_6"], OPHYS_4_6_DESCRIPTION))
    description_dict.update({"OPHYS_2": OPHYS_2_DESCRIPTION})
    description_dict.update({"OPHYS_5": OPHYS_5_DESCRIPTION})
    description_dict.update({"TRAINING_GRATINGS_0": TRAINING_GRATINGS_0})
    description_dict.update({"TRAINING_GRATINGS_1": TRAINING_GRATINGS_1})
    description_dict.update({"TRAINING_GRATINGS_2": TRAINING_GRATINGS_2})
    description_dict.update({"TRAINING_IMAGES_3": TRAINING_IMAGES_3})
    description_dict.update({"TRAINING_IMAGES_4": TRAINING_IMAGES_4})

    # Session type string will look something like: OPHYS_4_images_A
    n_match_str = 2
    if session_type.startswith("TRAINING"):
        n_match_str = 3
    truncated_session_type = "_".join(session_type.split("_")[:n_match_str])

    try:
        return description_dict[truncated_session_type]
    except KeyError as e:
        e_msg = (
            f"Encountered an unknown session type "
            f"({truncated_session_type}) when trying to determine "
            f"experiment descriptions. Valid session types are: "
            f"{description_dict.keys()}")
        raise RuntimeError(e_msg) from e


def get_task_parameters(data):

    task_parameters = {}
    task_parameters['blank_duration_sec'] = [float(x) for x in data["items"]["behavior"]['config']['DoC']['blank_duration_range']]
    task_parameters['stimulus_duration_sec'] = data["items"]["behavior"]['config']['DoC']['stimulus_window']
    task_parameters['omitted_flash_fraction'] = data["items"]["behavior"]['params'].get('omitted_flash_fraction', float('nan'))
    task_parameters['response_window_sec'] = [float(x) for x in data["items"]["behavior"]["config"]["DoC"]["response_window"]]
    task_parameters['reward_volume'] = data["items"]["behavior"]["config"]["reward"]["reward_volume"]
    task_parameters['stage'] = data["items"]["behavior"]["params"]["stage"]
    task_parameters['stimulus'] = next(iter(data["items"]["behavior"]["stimuli"]))
    task_parameters['stimulus_distribution'] = data["items"]["behavior"]["config"]["DoC"]["change_time_dist"]
    task_parameters['task'] = data["items"]["behavior"]["config"]["behavior"]["task_id"]
    n_stimulus_frames = 0
    for stim_type, stim_table in data["items"]["behavior"]["stimuli"].items():
        n_stimulus_frames += sum(stim_table.get("draw_log", []))
    task_parameters['n_stimulus_frames'] = n_stimulus_frames

    return task_parameters
