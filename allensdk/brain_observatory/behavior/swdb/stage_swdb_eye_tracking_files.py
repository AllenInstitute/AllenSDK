import os
import shutil
import json
import pandas as pd
from allensdk import one
from allensdk.internal.api import PostgresQueryMixin
import allensdk.brain_observatory.behavior.swdb.behavior_project_cache as bpc
from allensdk.internal.api import behavior_ophys_api as boa
cache_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/cache_20190813'

def get_eyetracking_file_path(ophys_experiment_id):
    api = PostgresQueryMixin()
    query = '''
    SELECT wkf.storage_directory || wkf.filename as full_path,
    wkf.filename
    FROM well_known_files wkf
    JOIN ophys_sessions os ON os.id=wkf.attachable_id
    JOIN ophys_experiments oe ON oe.ophys_session_id=os.id
    JOIN welL_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    WHERE wkft.name = 'EyeTracking Ellipses'
    AND oe.id = {};
    '''.format(ophys_experiment_id)
    #  return api.fetchone(query, strict=True)
    return pd.read_sql(query, api.get_connection())


def get_eyetracking_sync_times(ophys_experiment_id):
    '''
    I hate everything about this.
    '''
    eyetracking_file = get_eyetracking_file_path(ophys_experiment_id).iloc[0]
    pupil_data = pd.read_hdf(eyetracking_file['full_path'], key='pupil')
    api = boa.BehaviorOphysLimsApi(ophys_experiment_id)
    
    eyetracking_sync_times = api.get_sync_data()['eye_tracking']
    behaviormon_sync_times = api.get_sync_data()['behavior_monitoring']

    if pupil_data.shape[0] == eyetracking_sync_times.shape[0]:
        return pd.DataFrame({"timestamps":eyetracking_sync_times})
    elif pupil_data.shape[0] == behaviormon_sync_times.shape[0]:
        return pd.DataFrame({"timestamps":behaviormon_sync_times})
    else:
        raise ValueError("Eyetracking data did not match length of any sync time vectors")

if __name__=="__main__":
    output_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/eye_tracking_files'
    errors=0
    oeid_fn_mapping = {}
    cache = bpc.BehaviorProjectCache(cache_path)
    oeids = cache.experiment_table['ophys_experiment_id'].values
    for oeid in oeids:
        try:
            result = get_eyetracking_file_path(oeid).iloc[0]
            oeid_fn_mapping.update({int(oeid):result['filename']})
            print(result['full_path'])
            src = result['full_path']
            dest = os.path.join(output_dir, result['filename'])
            shutil.copyfile(src, dest)

            # Get the sync times and save them for this oeid
            # NOTE: Using experiment ID here, which we want to get away from, but it's what we use for this cache
            eye_tracking_sync_times = get_eyetracking_sync_times(oeid)
            et_sync_path = os.path.join(output_dir, 'eye_tracking_sync_ophys_experiment_{}.h5'.format(oeid))
            eye_tracking_sync_times.to_hdf(et_sync_path, key='df')
        except:
            errors+=1
            print("ERROR")

    output_json_path = os.path.join(output_dir, 'oeid_to_eye_tracking_file.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(oeid_fn_mapping, json_file)


