import pytest
import datetime
from pynwb import NWBFile#, NWBHDF5IO, TimeSeries
import os
import io
import pickle

@pytest.fixture(scope='function')
def nwbfile(tmpdir):

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    return nwbfile

@pytest.fixture(scope='function')
def tmpfilename(tmpdir):

    return os.path.join(str(tmpdir), 'test.nwb')

@pytest.fixture(scope='function')
def visbeh_pkl():

    bytes_io = io.BytesIO()
    D = {}
    D['items'] = {}
    D['items']['behavior'] = {}
    D['items']['behavior']["intervalsms"] = [16.6198723]*4
    D['items']['behavior']["update_count"] = 4
    D['start_time'] = datetime.datetime.now()
    D['platform_info'] = {}
    D['platform_info']['computer_name'] = 'test'
    D['items']['behavior']['stimuli'] = {}
    D["items"]["behavior"]["config"] = {}
    D["items"]["behavior"]["config"]["reward"] = {}
    D["items"]["behavior"]["config"]["reward"]["reward_volume"] = 0
    D["items"]["behavior"]["config"]["DoC"]={}
    D["items"]["behavior"]["config"]["DoC"]["auto_reward_volume"] = 0
    D["items"]["behavior"]["config"]["DoC"]["response_window"] = []
    D["items"]["behavior"]["config"]["DoC"]['stimulus_window'] = 0
    D["items"]["behavior"]["config"]["DoC"]['blank_duration_range'] = [0,0]
    D["items"]["behavior"]["config"]["DoC"]['change_time_dist'] = 'test'
    D["items"]["behavior"]["config"]["DoC"]['change_time_scale'] = 0
    D["items"]["behavior"]["config"]["DoC"]['warm_up_trials'] = 0
    D["items"]["behavior"]["config"]["DoC"]['failure_repeats'] = 0
    D["items"]["behavior"]["config"]["DoC"]['catch_freq'] = 0
    D["items"]["behavior"]["config"]["DoC"]['free_reward_trials'] = 0
    D["items"]["behavior"]["config"]["DoC"]['min_no_lick_time'] = 0
    D["items"]["behavior"]["config"]["DoC"]['max_task_duration_min'] = 0
    D["items"]["behavior"]["config"]["DoC"]['abort_on_early_response'] = 0
    D["items"]["behavior"]["config"]["DoC"]['initial_blank'] = 0
    D["items"]["behavior"]["config"]["DoC"]['periodic_flash'] = []
    D['items']['behavior']['lick_sensors'] = [{}]
    D['items']['behavior']['lick_sensors'][0]['lick_events'] = []
    D["items"]["behavior"]["params"] = {}
    D["items"]["behavior"]["params"]["task_id"] = 0
    D["items"]["behavior"]["config"]["behavior"] = {}
    D["items"]["behavior"]["config"]["behavior"]["volume_limit"] = 0
    D["items"]["behavior"]["trial_log"] = []
    D['items']['behavior']['encoders'] = [{}]
    D['items']['behavior']['encoders'][0]['dx'] = [ 1.397514e+02, -2.327280e-01,  4.668198e-02,  8.853951e-01,  7.473094e-01]
    D['items']['behavior']['encoders'][0]['vsig'] = [ 1.397514e+02, -2.327280e-01,  4.668198e-02,  8.853951e-01,  7.473094e-01]
    D['items']['behavior']['encoders'][0]['vin'] = [4.98354226257652, 4.987414049450308, 4.988704645074904, 4.982251666951925, 4.977089284453541]
    
    pickle.dump(D, bytes_io)

    return bytes_io
