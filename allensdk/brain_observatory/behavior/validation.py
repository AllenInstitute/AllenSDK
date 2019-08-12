import h5py
import os

from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.internal.api.ophys_lims_api import OphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession

class ValidationError(AssertionError):
    pass

def get_raw_ophys_file_shape(raw_filepath):
        with h5py.File(raw_filepath, 'r') as raw_file:
            raw_data_shape = raw_file['data'].shape
        return raw_data_shape

def validate_ophys_dff_length(ophys_experiment_id, api=None):
    api = OphysLimsApi() if api is None else api

    ophys_experiment_dir = api.get_ophys_experiment_dir(ophys_experiment_id)
    raw_filepath = os.path.join(ophys_experiment_dir, str(ophys_experiment_id)+'.h5')
    raw_data_shape = get_raw_ophys_file_shape(raw_filepath)

    dff_filepath = api.get_dff_file(ophys_experiment_id=ophys_experiment_id)
    dff_shape = get_raw_ophys_file_shape(dff_filepath)

    if raw_data_shape[0] != dff_shape[1]:
        raise ValidationError('dff length does not match raw data length')

def validate_ophys_timestamps(ophys_experiment_id, api=None):
    api = BehaviorOphysLimsApi() if api is None else api

    ophys_experiment_dir = api.get_ophys_experiment_dir(ophys_experiment_id)
    raw_filepath = os.path.join(ophys_experiment_dir, str(ophys_experiment_id)+'.h5')
    raw_data_shape = get_raw_ophys_file_shape(raw_filepath)

    ophys_timestamps_shape = api.get_ophys_timestamps(ophys_experiment_id=ophys_experiment_id).shape

    if raw_data_shape[0] != ophys_timestamps_shape[0]:
        raise ValidationError('ophys_timestamp length does not match raw data length')

def validate_last_trial_ends_adjacent_to_flash(ophys_experiment_id, api=None, verbose=False):
        # ensure that the last trial ends sometime on the last flash/blank cycle
        # i.e, if this is the last flash/blank iteration (high = stimulus present):
        #           -------           -------           -------           -------
        #          |       |         |       |         |       |         |       |
        # ---------         ---------         ---------         ---------         -------------------------------------------
        #                                                                ^                                                  ^
        # The last trial has to have ended somewhere between the two carrots where:
        #   the first carrot represents the time of the last recorded stimulus flash
        #   the second carrot represents the time at which another flash should have started, after accounting for the possibility of the session ending on an omitted flash
        
        api = BehaviorOphysLimsApi() if api is None else api
        session = BehaviorOphysSession(api)

        # get the flash/blank parameters
        max_flash_duration = session.stimulus_presentations['duration'].max()
        max_blank_duration = session.task_parameters['blank_duration_sec'][1]
        
        # count number of omitted flashes at the very end of the session
        N_final_omitted_flashes = session.stimulus_presentations.index.max() - session.stimulus_presentations.query('omitted == False').index.max()
        
        # get the start/end time of the last valid (non-omitted) flash
        last_flash_start = session.stimulus_presentations.query('omitted == False')['start_time'].iloc[-1]
        last_flash_end = session.stimulus_presentations.query('omitted == False')['stop_time'].iloc[-1]
        
        # calculate when the next stimulus should have flashed, after accounting for any omitted flashes at the end of the session
        next_flash_would_have_started = last_flash_end + max_blank_duration + N_final_omitted_flashes*(max_flash_duration + max_blank_duration)
        
        # get the end time of the last trial
        last_trial_end = session.trials.iloc[-1]['stop_time']
        
        if verbose:
            print('last flash ended at {}'.format(last_flash_end))
            print('another flash should have started by {}'.format(next_flash_would_have_started))
            print('last trial ended at {}'.format(last_trial_end))
        
        if not last_flash_start <= last_trial_end <= next_flash_would_have_started:
            raise ValidationError('The last trial does not end between the start of the last flash and the expected start time of the next flash')

if __name__ == "__main__":

    api = BehaviorOphysLimsApi()
    ophys_experiment_id_list = [775614751, 778644591, 787461073, 782675436, 783928214, 783927872,
                                787501821, 787498309, 788490510, 788488596, 788489531, 789359614,
                                790149413, 790709081, 791119849, 791453282, 791980891, 792813858,
                                792812544, 792816531, 792815735, 794381992, 794378505, 795076128,
                                795073741, 795952471, 795952488, 795953296, 795948257, 796106850,
                                796106321, 796108483, 796105823, 796308505, 797255551, 795075034,
                                798403387, 798404219, 799366517, 799368904, 799368262, 803736273,
                                805100431, 805784331, 805784313, 806456687, 806455766, 806989729,
                                807753318, 807752719, 807753334, 807753920, 796105304, 784482326,
                                779335436, 782675457, 791974731, 791979236,
                                800034837, 802649986, 806990245, 808621958,
                                808619526, 808619543, 808621034, 808621015]

    for ophys_experiment_id in ophys_experiment_id_list:
        validation_functions_to_run = [
            validate_ophys_timestamps, 
            validate_ophys_dff_length,
            validate_last_trial_ends_adjacent_to_flash,
        ]
        for validation_function in validation_functions_to_run:

            try:
                validation_function(ophys_experiment_id, api=api)
            except ValidationError as e:
                print(ophys_experiment_id, e)