import sys
import pandas as pd
import pytest
import numpy as np
import pytz
import datetime
import uuid
import os
import json


def pytest_ignore_collect(path, config):
    ''' The brain_observatory.ecephys submodule uses python 3.6 features that may not be backwards compatible!
    '''

    if sys.version_info < (3, 6):
        return True
    return False


@pytest.fixture
def running_data_df(running_speed):

    v_sig = np.ones_like(running_speed.values)
    v_in = np.ones_like(running_speed.values)
    dx = np.ones_like(running_speed.values)

    return pd.DataFrame({'speed': running_speed.values,
                         'dx': dx,
                         'v_sig': v_sig,
                         'v_in': v_in,
                         }, index=pd.Index(running_speed.timestamps, name='timestamps'))


@pytest.fixture
def stimulus_templates():

    image_template = np.zeros((3, 4, 5))
    image_template[1, :, :] = image_template[1, :, :] + 1
    image_template[2, :, :] = image_template[2, :, :] + 2

    return {'test1': image_template, 'test2': np.zeros((5, 2, 2))}


@pytest.fixture
def ophys_timestamps():
    return np.array([1., 2., 3.])


@pytest.fixture
def trials():
    return pd.DataFrame({
        'start_time': [1., 2., 4., 5., 6.],
        'stop_time': [2., 4., 5., 6., 8.],
        'a': [0.5, 0.4, 0.3, 0.2, 0.1],
        'b': [[], [1], [2, 2], [3], []],
        'c': ['a', 'bb', 'ccc', 'dddd', 'eeeee'],
        'd': [np.array([1]), np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])],
    }, index=pd.Index(name='trials_id', data=[0, 1, 2, 3, 4]))


@pytest.fixture
def licks():
    return pd.DataFrame({'time': [1., 2., 3.]})


@pytest.fixture
def rewards():
    return pd.DataFrame({'volume': [.01, .01, .01], 'autorewarded': [True, False, False]},
                        index=pd.Index(data=[1., 2., 3.], name='timestamps'))


@pytest.fixture
def image_api():
    from allensdk.brain_observatory.behavior.image_api import ImageApi
    return ImageApi


@pytest.fixture
def max_projection(image_api):
    return image_api.serialize(np.array([[1, 2], [3, 4]]), [.1, .1], 'mm')


@pytest.fixture
def average_image(max_projection):
    return max_projection


@pytest.fixture
def segmentation_mask_image(max_projection):
    return max_projection


@pytest.fixture
def stimulus_presentations_behavior(stimulus_templates, stimulus_presentations):

    image_sets = ['test1','test1', 'test1', 'test2', 'test2' ]
    stimulus_index_df = pd.DataFrame({'image_set': image_sets,
                                      'image_index': [0] * len(image_sets)},
                                       index=pd.Index(stimulus_presentations['start_time'], dtype=np.float64, name='timestamps'))

    df = stimulus_presentations.merge(stimulus_index_df, left_on='start_time', right_index=True)
    return df[sorted(df.columns)]


@pytest.fixture
def metadata():

    return {"ophys_experiment_id": 1234,
            "experiment_container_id": 5678,
            "ophys_frame_rate": 31.0,
            "stimulus_frame_rate": 60.0,
            "targeted_structure": "VISp",
            "imaging_depth": 375,
            "session_type": 'Unknown',
            "experiment_datetime": pytz.utc.localize(datetime.datetime.now()),
            "reporter_line": ["Ai93(TITL-GCaMP6f)"],
            "driver_line": ["Camk2a-tTA", "Slc17a7-IRES2-Cre"],
            "LabTracks_ID": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_session_uuid": uuid.uuid4(),
            "emission_lambda": 1.0,
            "excitation_lambda": 1.0,
            "indicator": 'HW',
            "field_of_view_width": 2,
            "field_of_view_height": 2,
            "rig_name": 'my_device',
            "sex": 'M',
            "age": 'P139',
            }


@pytest.fixture
def task_parameters():

    return {"blank_duration_sec": [0.5, 0.5],
            "stimulus_duration_sec": 6000.0,
            "omitted_flash_fraction": float('nan'),
            "response_window_sec": [0.15, 0.75],
            "reward_volume": 0.007,
            "stage": "OPHYS_6_images_B",
            "stimulus": "images",
            "stimulus_distribution": "geometric",
            "task": "DoC_untranslated",
            "n_stimulus_frames": 69882
            }


@pytest.fixture
def cell_specimen_table():
    return pd.DataFrame({'cell_roi_id': [123, 321],
                         'x': [1, 1],
                         'y': [1, 1],
                         'width': [1, 1],
                         'height': [1, 1],
                         'valid_roi':[True, False],
                         'max_correction_up':[1., 1.],
                         'max_correction_down':[1., 1.],
                         'max_correction_left':[1., 1.],
                         'max_correction_right':[1., 1.],
                         'mask_image_plane':[1, 1],
                         'ophys_cell_segmentation_run_id':[1, 1],
                         'image_mask': [np.array([[True, True], [False, False]]), np.array([[True, True], [False, False]])]},
                          index=pd.Index([None, None], dtype=int, name='cell_specimen_id'))


@pytest.fixture
def dff_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'dff': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def corrected_fluorescence_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'corrected_fluorescence': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def motion_correction(ophys_timestamps):
    return pd.DataFrame({'x': np.ones_like(ophys_timestamps),
                         'y': np.ones_like(ophys_timestamps)})


@pytest.fixture
def session_data():

    data = {'ophys_experiment_id': 789359614,
            'surface_2p_pixel_size_um': 0.78125,
            "max_projection_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_a13a.png",
            "sync_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_sync.h5",
            "rig_name": "CAM2P.5",
            "movie_width": 447,
            "movie_height": 512,
            "container_id": 814796558,
            "targeted_structure": "VISp",
            "targeted_depth": 375,
            "stimulus_name": "Unknown",
            "date_of_acquisition": '2018-11-30 23:28:37',
            "reporter_line": ["Ai93(TITL-GCaMP6f)"],
            "driver_line": ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
            "external_specimen_name": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_stimulus_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl",
            "dff_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/789359614_dff.h5",
            "ophys_cell_segmentation_run_id": 789410052,
            "cell_specimen_table_dict": json.load(open(os.path.join(os.path.dirname(__file__), 'cell_specimen_table_789359614.json'), 'r')),
            "demix_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/demix/789359614_demixed_traces.h5",
            "average_intensity_projection_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/avgInt_a1X.png",
            "rigid_motion_transform_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/789359614_rigid_motion_transform.csv",
            "segmentation_mask_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_masks.tif",
            "sex": "F",
            "age": "P139"}

    return data


@pytest.fixture
def rylan_dprime_meta():
    # * we use the result of rylan's code as a value test against our refactored version
    # * all code is embedded within to try to prevent some side effects cheaply

    from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
    from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession

    import pandas as pd
    import numpy as np


    def get_response_rates(session, sliding_window=100):
        '''Calculates response rates for hits, catches, and d-prime across a rolling window.
        Additionally checks that licks are present in the correct response window for the correct hits (go trials)
        and catch false alarms.
    
        Parameters
        ----------
        session : AllenSDK session object 
            session object for an individual experiment
        sliding_window : int
            sliding window (in data points) to calculate rolling d-prime, hit-rate, and false-alarm over
        
        Returns
        ------
        hit_rate (array),
        catch_rate (array),
        dprime (array)
        '''

        #go responses: for this we check if the trial is go and the mouse was rewarded
        #---------------------------------------------
        go_responses = pd.Series([np.nan] * len(session.trials))
        go_trials=session.trials.loc[(session.trials['go']==True) & (session.trials['auto_rewarded']!=True) 
            & (session.trials['reward_times'])]

        #we want to have a check that ensures that the definition of go trials above are classified as hits and that
        #there is a lick in the response window

        response_win_start=session.task_parameters['response_window_sec'][0]
        response_win_end=session.task_parameters['response_window_sec'][1]

        licks_present_go=check_lick_in_resp_wind(trials=go_trials,
                                    window_start=response_win_start,
                                    window_end=response_win_end)

        # ensure that all hit trials have a lick present in the window
        assert(set(go_trials['hit'])==set(licks_present_go))

        go_responses[session.trials.loc[(session.trials['go']==True) & (session.trials['auto_rewarded']!=True) 
                        & (session.trials['reward_times'])].index]=1

        go_responses[session.trials.loc[(session.trials['go']==True) & (session.trials['auto_rewarded']!=True) 
                        & (session.trials['reward_times'].str.len() == 0)].index]=0

        #note that after this, trials that are not GO are encoded as NaN in the go responses mask
        hit_rate = go_responses.rolling(window=sliding_window,min_periods=0).mean()


        #catch responses: for this we check if false alarm and catch are true
        #----------------------------------------------------------------------
        #ideally future versions would check the correct reject logic
        catch_responses = pd.Series([np.nan] * len(session.trials))

        catch_fa_trials=session.trials.loc[(session.trials['catch']==True) & (session.trials['false_alarm']==True)]

        catch_responses[session.trials.loc[(session.trials['catch']==True) & (session.trials['false_alarm']==True)].index]=1

        #we assert that for a catch false alarm, there must be a lick in the response window on a catch trial
        licks_present_catch=check_lick_in_resp_wind(trials=catch_fa_trials,
                                                window_start=response_win_start,
                                                window_end=response_win_end)

        assert(set(catch_fa_trials['false_alarm'])==set(licks_present_catch))

        catch_responses[session.trials.loc[(session.trials['catch']==True) & (session.trials['false_alarm']==False)].index]=0

        catch_rate = catch_responses.rolling(window=sliding_window,min_periods=0).mean()

        #calculate d-prime using previously defined function
        d_prime = dprime(hit_rate, catch_rate)

        return hit_rate.values, catch_rate.values, d_prime


    def dprime(hit_rate, fa_rate, limits=(0.01, 0.99)):

        from scipy.stats import norm
        """ calculates the d-prime for a given hit rate and false alarm rate
        https://en.wikipedia.org/wiki/Sensitivity_index
        Parameters
        ----------
        hit_rate : float
            rate of hits in the True class
        fa_rate : float
            rate of false alarms in the False class
        limits : tuple, optional
            limits on extreme values, which distort. default: (0.01,0.99)
        Returns
        -------
        d_prime
        """
        assert limits[0] > 0.0, 'limits[0] must be greater than 0.0'
        assert limits[1] < 1.0, 'limits[1] must be less than 1.0'
        Z = norm.ppf

        # Limit values in order to avoid d' infinity
        hit_rate = np.clip(hit_rate, limits[0], limits[1])
        fa_rate = np.clip(fa_rate, limits[0], limits[1])

        return Z(hit_rate) - Z(fa_rate)


    def check_lick_in_resp_wind(trials,window_start,window_end):
        '''Returns whether there is a lick time in the passed window (start to end)
        
        Parameters
            ----------
            trials : dataframe 
                a trials Dataframe extracted from the AllenSDK session object, or a portion there of
                
        Returns
            ------
            Boolean array of whether there was a lick in the passed window (np.array)
        '''

        trials_cop=trials.copy()
        #response time is relative to the image change
        trials_cop['start_response_window']=trials_cop['change_time']+window_start

        trials_cop['end_response_window']=trials_cop['change_time']+window_end

        trials_cop['lick_in_window']=trials_cop.apply(lambda x: any(x['start_response_window']< i< x['end_response_window'] for i in x['lick_times']), 1)

        return np.array(trials_cop['lick_in_window'])


    ophys_experiment_id = 820307518

    # Load data from LIMS direction:
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysSession(api)

    hit_rate,catch_rate,d_prime=get_response_rates(session, )
    return d_prime, hit_rate, catch_rate, ophys_experiment_id, 