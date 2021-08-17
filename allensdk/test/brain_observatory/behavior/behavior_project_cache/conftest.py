import pytest
import pandas as pd
import io
import semver

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.\
        data_io.behavior_project_cloud_api import MANIFEST_COMPATIBILITY


@pytest.fixture
def s3_cloud_cache_data():

    all_versions = {}
    all_versions['data'] = {}
    all_versions['metadata'] = {}

    min_compat = semver.parse_version_info(MANIFEST_COMPATIBILITY[0])
    versions = []

    version = str(min_compat)
    versions.append(version)
    data = {}
    metadata = {}

    data['ophys_file_1.nwb'] = {'file_id': 1,
                                'data': b'abcde'}

    data['ophys_file_2.nwb'] = {'file_id': 2,
                                'data': b'fghijk'}

    data['behavior_file_3.nwb'] = {'file_id': 3,
                                   'data': b'12345'}

    data['behavior_file_4.nwb'] = {'file_id': 4,
                                   'data': b'67890'}

    o_session = [{'ophys_session_id': 111,
                  'file_id': 1},
                 {'ophys_session_id': 222,
                  'file_id': 2}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_session_table'] = bytes(buff.read(), 'utf-8')

    b_session = [{'behavior_session_id': 333,
                  'file_id': 3,
                  'species': 'mouse'},
                 {'behavior_session_id': 444,
                  'file_id': 4,
                  'species': 'mouse'}]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['behavior_session_table'] = bytes(buff.read(), 'utf-8')

    o_session = [{'ophys_experiment_id': 5111,
                  'file_id': 1},
                 {'ophys_experiment_id': 5222,
                  'file_id': 2}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_experiment_table'] = bytes(buff.read(), 'utf-8')

    o_cells = {
            'cell_roi_id': {0: 9080884343, 1: 1080884173, 2: 1080883843},
            'cell_specimen_id': {0: 1086496928, 1: 1086496914, 2: 1086496838},
            'ophys_experiment_id': {0: 775614751, 1: 775614751, 2: 775614751}}
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_cells_table'] = bytes(buff.read(), 'utf-8')

    all_versions['data'][version] = data
    all_versions['metadata'][version] = metadata

    version = str(min_compat.bump_minor())
    versions.append(version)
    data = {}
    metadata = {}

    data['ophys_file_1.nwb'] = {'file_id': 1,
                                'data': b'lmnopqrs'}

    data['ophys_file_2.nwb'] = {'file_id': 2,
                                'data': b'fghijk'}

    data['behavior_file_3.nwb'] = {'file_id': 3,
                                   'data': b'12345'}

    data['behavior_file_4.nwb'] = {'file_id': 4,
                                   'data': b'67890'}

    data['ophys_file_5.nwb'] = {'file_id': 5,
                                'data': b'98765'}

    o_session = [{'ophys_session_id': 222,
                  'file_id': 1},
                 {'ophys_session_id': 333,
                  'file_id': 2}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_session_table'] = bytes(buff.read(), 'utf-8')

    b_session = [{'behavior_session_id': 777,
                  'file_id': 3,
                  'species': 'mouse'},
                 {'behavior_session_id': 888,
                  'file_id': 4,
                  'species': 'mouse'}]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['behavior_session_table'] = bytes(buff.read(), 'utf-8')

    o_session = [{'ophys_experiment_id': 5444,
                  'file_id': 1},
                 {'ophys_experiment_id': 5666,
                  'file_id': 2},
                 {'ophys_experiment_id': 5777,
                  'file_id': 5}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_experiment_table'] = bytes(buff.read(), 'utf-8')

    o_cells = {
            'cell_roi_id': {0: 1080884343, 1: 1080884173, 2: 1080883843},
            'cell_specimen_id': {0: 1086496928, 1: 1086496914, 2: 1086496838},
            'ophys_experiment_id': {0: 775614751, 1: 775614751, 2: 775614751}}
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_cells_table'] = bytes(buff.read(), 'utf-8')

    all_versions['data'][version] = data
    all_versions['metadata'][version] = metadata

    return all_versions, versions


@pytest.fixture
def data_update():
    data = {}
    metadata = {}

    data['ophys_file_1.nwb'] = {'file_id': 1,
                                'data': b'11235'}

    data['ophys_file_2.nwb'] = {'file_id': 2,
                                'data': b'8132134'}

    data['behavior_file_3.nwb'] = {'file_id': 3,
                                   'data': b'04916'}

    data['behavior_file_4.nwb'] = {'file_id': 4,
                                   'data': b'253649'}

    data['ophys_file_5.nwb'] = {'file_id': 5,
                                'data': b'98765'}

    o_session = [{'ophys_session_id': 1110,
                  'file_id': 1},
                 {'ophys_session_id': 2220,
                  'file_id': 2}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_session_table'] = bytes(buff.read(), 'utf-8')

    b_session = [{'behavior_session_id': 3330,
                  'file_id': 3,
                  'species': 'mouse'},
                 {'behavior_session_id': 4440,
                  'file_id': 4,
                  'species': 'mouse'}]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['behavior_session_table'] = bytes(buff.read(), 'utf-8')

    o_session = [{'ophys_experiment_id': 6111,
                  'file_id': 1},
                 {'ophys_experiment_id': 6222,
                  'file_id': 2},
                 {'ophys_experiment_id': 63456,
                  'file_id': 5}]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_experiment_table'] = bytes(buff.read(), 'utf-8')

    o_cells = {
            'cell_roi_id': {0: 9080884343, 1: 1080884173, 2: 1080883843},
            'cell_specimen_id': {0: 1086496928, 1: 1086496914, 2: 1086496838},
            'ophys_experiment_id': {0: 775614751, 1: 775614751, 2: 775614751}}
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata['ophys_cells_table'] = bytes(buff.read(), 'utf-8')

    return {'data': data, 'metadata': metadata}
