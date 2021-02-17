import re

description_dict = {
    # key is a regex and value is returned on match
    r"\AOPHYS_0_images": "A behavior training session performed on the 2-photon calcium imaging setup but without recording neural activity, with the goal of habituating the mouse to the experimental setup before commencing imaging of neural activity. Habituation sessions are change detection with the same image set on which the mouse was trained. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_[1|3]_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse performs a visual change detection task with a set of natural images upon which it has been previously trained. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_2_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse is passively shown changes in natural scene images upon which it was previously trained as the change detection task is played in open loop mode, with the lick-response sensory withdrawn and the mouse is unable to respond to changes or receive reward feedback. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_[4|6]_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse performs a visual change detection task with natural scene images that are unique from those on which the mouse was trained prior to the imaging phase of the experiment. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_5_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse is passively shown changes in natural scene images that are unique from those on which the mouse was trained prior to the imaging phase of the experiment. In this session, the change detection task is played in open loop mode, with the lick-response sensory withdrawn and the mouse is unable to respond to changes or receive reward feedback. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\ATRAINING_0_gratings": "An associative training session where a mouse is automatically rewarded when a grating stimulus changes orientation. Grating stimuli are  full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree, with orientation changes between 0 and 90 degrees, at two spatial phases. Delivered rewards are 5ul in volume, and the session lasts for 15 minutes.",  # noqa: E501
    r"\ATRAINING_1_gratings": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of  full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree. Orientation changes between 0 and 90 degrees occur with no intervening gray period. Delivered rewards are 10ul in volume, and the session lasts 60 minutes",  # noqa: E501
    r"\ATRAINING_2_gratings": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree. Gratings of 0 or 90 degrees are presented for 250 ms with a 500 ms intervening gray period. Delivered rewards are 10ul in volume, and the session lasts 60 minutes.",  # noqa: E501
    r"\ATRAINING_3_images": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 10ul in volume, and the session lasts for 60 minutes",  # noqa: E501
    r"\ATRAINING_4_images": "An operant behavior training session where a mouse must lick a spout following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 7ul in volume, and the session lasts for 60 minutes",  # noqa: E501
    r"\ATRAINING_5_images": "An operant behavior training session where a mouse must lick a spout following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 7ul in volume. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session."  # noqa: E501
    }


def get_expt_description(session_type: str) -> str:
    """Determine a behavior ophys session's experiment description based on
    session type. Matches the regex patterns defined as the keys in
    description_dict

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
    match = dict()
    for k, v in description_dict.items():
        if re.match(k, session_type) is not None:
            match.update({k: v})

    if len(match) != 1:
        emsg = (f"session type should match one and only one possible pattern "
                f"template. '{session_type}' matched {len(match)} pattern "
                "templates.")
        if len(match) > 1:
            emsg += f"{list(match.keys())}"
        emsg += f"the regex pattern templates are {list(description_dict)}"
        raise RuntimeError(emsg)

    return match.popitem()[1]


def get_task_parameters(data):
    behavior = data["items"]["behavior"]

    task_parameters = {}
    task_parameters['blank_duration_sec'] = \
        [float(x) for x in behavior['config']['DoC']['blank_duration_range']]
    task_parameters['stimulus_duration_sec'] = \
        behavior['config']['DoC']['stimulus_window']
    task_parameters['omitted_flash_fraction'] = \
        behavior['params'].get('flash_omit_probability', float('nan'))
    task_parameters['response_window_sec'] = \
        [float(x) for x in behavior["config"]["DoC"]["response_window"]]
    task_parameters['reward_volume'] = \
        behavior["config"]["reward"]["reward_volume"]
    task_parameters['stage'] = behavior["params"]["stage"]
    task_parameters['stimulus'] = next(iter(behavior["stimuli"]))
    task_parameters['stimulus_distribution'] = \
        behavior["config"]["DoC"]["change_time_dist"]
    task_parameters['task'] = behavior["config"]["behavior"]["task_id"]
    n_stimulus_frames = 0
    for stim_type, stim_table in behavior["stimuli"].items():
        n_stimulus_frames += sum(stim_table.get("draw_log", []))
    task_parameters['n_stimulus_frames'] = n_stimulus_frames

    return task_parameters
