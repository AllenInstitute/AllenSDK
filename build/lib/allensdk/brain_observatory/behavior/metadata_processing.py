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