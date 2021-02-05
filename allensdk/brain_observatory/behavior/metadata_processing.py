OPHYS_1_3_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse "
    "brain as the mouse performs a visual change detection task "
    "with a set of natural scenes upon which it has previously been "
    "trained."
)
OPHYS_2_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the "
    "mouse brain as the mouse is shown images from a "
    "change detection task with a set of natural scenes "
    "upon which it has previously been trained, but with "
    "the lick-response sensor withdrawn (passive/open "
    "loop mode)."
)
OPHYS_4_6_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse "
    "brain as the mouse performs a visual change detection task "
    "with a set of natural scenes that are unique from those on "
    "which it was previously trained."
)
OPHYS_5_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the "
    "mouse brain as the mouse is shown images from a "
    "change detection task with a set of natural scenes "
    "that are unique from those on which it was "
    "previously trained, but with the lick-response "
    "sensor withdrawn (passive/open loop mode)."
)
TRAINING_DESCRIPTION = (
    "A training session where a mouse performs a visual change detection task "
    "with a set of natural scenes. Successfully completing the task delivers "
    "a water reward via a lick spout."
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
    ophys_1_3 = dict.fromkeys(["OPHYS_1", "OPHYS_3"], OPHYS_1_3_DESCRIPTION)
    ophys_4_6 = dict.fromkeys(["OPHYS_4", "OPHYS_6"], OPHYS_4_6_DESCRIPTION)
    ophys_2_5 = {"OPHYS_2": OPHYS_2_DESCRIPTION,
                 "OPHYS_5": OPHYS_5_DESCRIPTION}
    training = dict.fromkeys(
        ["TRAINING_1", "TRAINING_2", "TRAINING_3", "TRAINING_4", "TRAINING_5"],
        TRAINING_DESCRIPTION)

    expt_description_dict = {**ophys_1_3, **ophys_2_5, **ophys_4_6, **training}

    # Session type string will look something like: OPHYS_4_images_A
    truncated_session_type = "_".join(session_type.split("_")[:2])

    try:
        return expt_description_dict[truncated_session_type]
    except KeyError as e:
        e_msg = (
            f"Encountered an unknown session type "
            f"({truncated_session_type}) when trying to determine "
            f"experiment descriptions. Valid session types are: "
            f"{expt_description_dict.keys()}")
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
