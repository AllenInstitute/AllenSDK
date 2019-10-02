import os

from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession


basedir = '/allen/aibs/technology/nicholasc/behavior_ophys'

rel_filepath_list = ['behavior_ophys_session_805784331.nwb',
                     'behavior_ophys_session_789359614.nwb',
                     'behavior_ophys_session_803736273.nwb',
                     'behavior_ophys_session_808621958.nwb',
                     'behavior_ophys_session_795948257.nwb']

for rel_filepath in rel_filepath_list:

    full_filepath = os.path.join(basedir, rel_filepath)
    session = BehaviorOphysSession(api=BehaviorOphysNwbApi(full_filepath))
    print(session.metadata['ophys_experiment_id'])
