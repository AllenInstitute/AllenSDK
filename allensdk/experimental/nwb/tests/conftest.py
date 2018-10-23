import pytest
import datetime
from pynwb import NWBFile, NWBHDF5IO
import os
import io
import pickle
import numpy as np

from allensdk.experimental.nwb.tests.utilities.comparators import generic_assert_equal


@pytest.fixture
def vb_pkl():
    return '/allen/programs/braintv/production/visualbehavior/prod0/specimen_710269829/ophys_session_759332825/759332825_stim.pkl'


@pytest.fixture
def vb_sync():
    return '/allen/programs/braintv/production/visualbehavior/prod0/specimen_710269829/ophys_session_759332825/759332825_sync.h5'


@pytest.fixture
def vc_pkl():
    return '/allen/programs/braintv/production/neuralcoding/prod53/specimen_691654617/ophys_session_714254764/714254764_388801_20180626_stim.pkl'


@pytest.fixture
def vc_sync():
    return '/allen/programs/braintv/production/neuralcoding/prod53/specimen_691654617/ophys_session_714254764/714254764_388801_20180626_sync.h5'


@pytest.fixture
def nwb_filename(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test").join("test.nwb")
    return str(nwb)

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
def behaviorimagesfilename(tmpdir):

    image_dict = {}
    image_dict['im065'] = {}
    image_dict['im077'] = {}
    image_dict['im065']['im065'] = np.zeros((3,6))
    image_dict['im077']['im077'] = np.ones((3,6))
    
    fname = os.path.join(str(tmpdir), 'images.pkl')
    pickle.dump(image_dict, open(fname, 'w'))

    return fname

@pytest.fixture(scope='function')
def visbeh_pkl(behaviorimagesfilename):

    bytes_io = io.BytesIO()
    D = {}
    D['items'] = {}
    D['items']['behavior'] = {}
    D['items']['behavior']["intervalsms"] = [16.6198723]*5
    D['items']['behavior']["update_count"] = 4
    D['start_time'] = datetime.datetime.now()
    D['platform_info'] = {}
    D['platform_info']['computer_name'] = 'test'
    D['items']['behavior']['stimuli'] = {}
    D['items']['behavior']['stimuli']['images'] = {}
    D['items']['behavior']['stimuli']['images']['size'] = (1174, 918)
    D['items']['behavior']['stimuli']['images']['obj_type'] = 'DoCImageStimulus'
    D['items']['behavior']['stimuli']['images']['sampling'] = 'even'
    D['items']['behavior']['stimuli']['images']['set_log'] = [('Image', 'im065', .05, 0),('Image', 'im077', 26.01941239779431, 2)]
    D['items']['behavior']['stimuli']['images']['change_log'] = [(('im065', 'im065'), ('im077', 'im077'), 26.019579445142398, 2)]
    D['items']['behavior']['stimuli']['images']['draw_log'] = [0,1,0,1,0,0]
    D['items']['behavior']['stimuli']['images']['image_path'] = behaviorimagesfilename
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
    D['items']['behavior']['encoders'][0]['dx'] = [ 1.397514e+02, -2.327280e-01,  4.668198e-02,  8.853951e-01,  7.473094e-01,  7.473094e-01]
    D['items']['behavior']['encoders'][0]['vsig'] = [ 1.397514e+02, -2.327280e-01,  4.668198e-02,  8.853951e-01,  7.473094e-01,  7.473094e-01]
    D['items']['behavior']['encoders'][0]['vin'] = [4.98354226257652, 4.987414049450308, 4.988704645074904, 4.982251666951925, 4.977089284453541, 4.977089284453541]
    
    pickle.dump(D, bytes_io)

    return bytes_io


@pytest.fixture
def roundtripper(tmpdir_factory):
    def roundtripper(nwbfile, file_name='test.nwb'):
        temp_dir = str(tmpdir_factory.mktemp('roundtrip'))
        file_path = os.path.join(temp_dir, file_name)

        with NWBHDF5IO(file_path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        with NWBHDF5IO(file_path, 'r') as nwb_file_reader:
            obtained = nwb_file_reader.read()
            generic_assert_equal(nwbfile, obtained)
    return roundtripper
