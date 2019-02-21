import pandas as pd

def get_task_parameters(core_data):
    task_parameters = {}
    task_parameters['blank_duration'] = core_data['metadata']['blank_duration_range'][0]
    task_parameters['stimulus_duration'] = core_data['metadata']['stim_duration']
    if 'omitted_flash_fraction' in core_data['metadata']['params'].keys():
        task_parameters['omitted_flash_fraction'] = core_data['metadata']['params']['omitted_flash_fraction']
    else:
        task_parameters['omitted_flash_fraction'] = None
    task_parameters['response_window'] = core_data['metadata']['response_window']
    task_parameters['reward_volume'] = core_data['metadata']['rewardvol']
    task_parameters['stage'] = core_data['metadata']['stage']
    task_parameters['stimulus'] = core_data['metadata']['stimulus']
    task_parameters['stimulus_distribution'] = core_data['metadata']['stimulus_distribution']
    task_parameters['task'] = core_data['metadata']['task']
    task_parameters['n_stimulus_frames'] = core_data['metadata']['n_stimulus_frames']

    return task_parameters