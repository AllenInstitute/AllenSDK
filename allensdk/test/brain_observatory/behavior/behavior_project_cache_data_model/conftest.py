import os
import copy
import numpy as np
import pytest
import pandas as pd
import tempfile

from allensdk.brain_observatory.behavior.behavior_project_cache \
    import VisualBehaviorOphysProjectCache

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    tables.util.experiments_table_utils import (
        add_experience_level_to_experiment_table,
        add_passive_flag_to_ophys_experiment_table,
        add_image_set_to_experiment_table)

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import \
    get_prior_exposures_to_session_type, \
    get_prior_exposures_to_image_set, \
    get_prior_exposures_to_omissions
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.reporter_line import \
    ReporterLine


@pytest.fixture(scope='session')
def behavior_session_id_list():
    """
    List of behavior_session_id; the most fundamental fixture
    """
    return list(range(1, 9))


@pytest.fixture(scope='session')
def session_name_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to session_name
    """
    return {ii: f'session_{ii}'
            for ii in behavior_session_id_list}


@pytest.fixture(scope='session')
def date_of_acquisition_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to date of acquisition
    """
    return {ii: np.datetime64(f'2020-02-{ii:02d}')
            for ii in behavior_session_id_list}


@pytest.fixture(scope='session')
def session_type_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to session_type
    """
    rng = np.random.default_rng(871231)
    possible = ('TRAINING_1_gratings',
                'OPHYS_1_images_A',
                'OPHYS_1_images_B')

    vals = rng.choice(possible,
                      size=len(behavior_session_id_list),
                      replace=True)

    return {ii: vv
            for ii, vv in zip(behavior_session_id_list,
                              vals)}


@pytest.fixture(scope='session')
def project_code_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to project_code
    """
    return {ii: 'code{ii}'
            for ii in behavior_session_id_list}


@pytest.fixture(scope='session')
def specimen_id_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to specimen_id
    """
    return {ii: 1111*ii
            for ii in behavior_session_id_list}


@pytest.fixture(scope='session')
def genotype_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to full_genotype
    """
    rng = np.random.default_rng(981232)
    possible = ('foo-SlcCre',
                'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt',
                'bar',
                'foobar')
    chosen = rng.choice(possible,
                        size=len(behavior_session_id_list),
                        replace=True)
    return {ii: val
            for ii, val in zip(behavior_session_id_list, chosen)}


@pytest.fixture(scope='session')
def reporter_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to reporter_line
    """
    return {ii: f"Ai{90+ii}(TITL-GCaMP6f)"
            for ii in behavior_session_id_list}


@pytest.fixture(scope='session')
def driver_lookup(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to driver_line.
    Note: driver_line is a list of strings
    """
    rng = np.random.default_rng(1723213)
    possible = (["aa"],
                ["aa", "bb"],
                ["cc"],
                ["cc", "dd"])
    chosen = rng.choice(possible,
                        size=len(behavior_session_id_list),
                        replace=True)
    return {ii: val
            for ii, val in zip(behavior_session_id_list,
                               chosen)}


@pytest.fixture(scope='session')
def behavior_session_data_fixture(behavior_session_id_list,
                                  session_name_lookup,
                                  date_of_acquisition_lookup,
                                  session_type_lookup,
                                  specimen_id_lookup,
                                  genotype_lookup,
                                  reporter_lookup,
                                  driver_lookup):
    """
    List of dicts. Each dict is an entry in the raw
    behavior_session_table as would be returned by the
    fetch_api
    """

    behavior_session_list = []
    for s_id in behavior_session_id_list:

        genotype = genotype_lookup[s_id]
        driver = driver_lookup[s_id]
        reporter = reporter_lookup[s_id]
        date = date_of_acquisition_lookup[s_id]
        specimen_id = specimen_id_lookup[s_id]
        s_name = session_name_lookup[s_id]
        s_type = session_type_lookup[s_id]
        datum = {'behavior_session_id': s_id,
                 'session_name': s_name,
                 'date_of_acquisition': date,
                 'specimen_id': specimen_id,
                 'session_type': s_type,
                 'equipment_name': 'MESO2.0',
                 'donor_id': 20+s_id,
                 'full_genotype': genotype,
                 'sex': ['m', 'f'][s_id % 2],
                 'age_in_days': s_id*7,
                 'foraging_id': s_id+30,
                 'mouse_id': s_id+40,
                 'reporter_line': reporter,
                 'driver_line': driver}

        behavior_session_list.append(datum)

    return behavior_session_list


@pytest.fixture()
def behavior_session_table(behavior_session_data_fixture):
    """
    The behavior_session_table dataframe as returned by the
    fetch_api
    """
    data = []
    index = []
    for datum in behavior_session_data_fixture:
        datum = copy.deepcopy(datum)
        index.append(datum.pop('behavior_session_id'))
        data.append(datum)

    df = pd.DataFrame(
              data,
              index=pd.Index(index, name='behavior_session_id'))
    return df


@pytest.fixture(scope='session')
def behavior_session_to_ophys_session_map(behavior_session_id_list):
    """
    Dict mapping behavior_session_id to ophys_session_id.
    This is a one-to-one mapping, though not all behavior_sessions
    have corresponding ophys_sessions
    """
    lookup = dict()
    ophys_id = 88
    for ii in range(0, len(behavior_session_id_list)):
        if ii % 3 == 0:
            continue
        lookup[behavior_session_id_list[ii]] = ophys_id
        ophys_id += 1
    return lookup


@pytest.fixture(scope='session')
def ophys_session_to_experiment_map(behavior_session_to_ophys_session_map):
    """
    Dict mapping ophys_session_id to a list of ophys_experiment_ids
    (this is a one-to-many relationship)
    """
    lookup = dict()
    i0 = 1000
    dd = 5
    ophys_vals = list(behavior_session_to_ophys_session_map.values())
    ophys_vals.sort()
    for ii in ophys_vals:
        lookup[ii] = list(range(i0, i0+dd))
        i0 += dd

    return lookup


@pytest.fixture(scope='session')
def ophys_experiment_to_container_map(ophys_session_to_experiment_map):
    """
    Dict mapping ophys_experiment_id to a list of ophys_container_ids
    (this is a one-to-many relationship)
    """
    lookup = dict()
    container_id = 4000
    key_list = list(ophys_session_to_experiment_map.keys())
    key_list.sort()
    for key in key_list:
        experiment_list = ophys_session_to_experiment_map[key]
        for exp_id in experiment_list:
            local_list = []
            for ii in range(7):
                container_id += 1
                local_list.append(container_id)
            lookup[exp_id] = local_list
    return lookup


@pytest.fixture(scope='session')
def container_state_lookup(ophys_experiment_to_container_map):
    """
    Dict mapping ophys_container_id to container_workflow_state
    Note: each ophys_experiment_id can only be associated with
    one 'published' ophys_container.
    """
    rng = np.random.default_rng(66232)
    exp_id_list = list(ophys_experiment_to_container_map.keys())
    exp_id_list.sort()
    lookup = dict()
    for exp_id in exp_id_list:
        local_container_list = ophys_experiment_to_container_map[exp_id]
        for container_id in local_container_list:
            assert container_id not in lookup
            lookup[container_id] = 'junk'
        good_container = rng.choice(local_container_list)
        lookup[good_container] = 'published'
    return lookup


@pytest.fixture(scope='session')
def experiment_state_lookup(ophys_session_data_fixture):
    """
    Dict mapping ophys_experiment_id to the experiment_workflow_state
    """
    rng = np.random.default_rng(772312)
    exp_id_list = []
    for datum in ophys_session_data_fixture:
        for exp_id in datum['ophys_experiment_id']:
            if exp_id not in exp_id_list:
                exp_id_list.append(exp_id)
    lookup = dict()
    for exp_id in exp_id_list:
        lookup[exp_id] = ['passed', 'failed'][rng.integers(0, 2)]
    return lookup


@pytest.fixture(scope='session')
def ophys_session_data_fixture(project_code_lookup,
                               session_name_lookup,
                               date_of_acquisition_lookup,
                               specimen_id_lookup,
                               session_type_lookup,
                               ophys_session_to_experiment_map,
                               ophys_experiment_to_container_map,
                               behavior_session_to_ophys_session_map):
    """
    List of dicts.
    Each dict is one entry in the ophys_session_table as returned
    by the fetch_api.
    """

    ophys_session_list = []
    ophys_session_id_list = list(ophys_session_to_experiment_map.keys())
    ophys_session_id_list.sort()
    for beh in behavior_session_to_ophys_session_map:
        o_session = behavior_session_to_ophys_session_map[beh]
        container_list = []
        for exp_id in ophys_session_to_experiment_map[o_session]:
            container_list += ophys_experiment_to_container_map[exp_id]

        datum = {'behavior_session_id': beh,
                 'project_code': project_code_lookup[beh],
                 'date_of_acquisition': date_of_acquisition_lookup[beh],
                 'session_name': session_name_lookup[beh],
                 'session_type': session_type_lookup[beh],
                 'ophys_experiment_id':
                 ophys_session_to_experiment_map[o_session],
                 'ophys_container_id': container_list,
                 'specimen_id': 9*beh,
                 'ophys_session_id': o_session}

        ophys_session_list.append(datum)
    return ophys_session_list


@pytest.fixture()
def ophys_session_table(ophys_session_data_fixture):
    """
    The ophys_session_table dataframe as returned by the fetch_api
    """
    data = []
    index = []
    for datum in ophys_session_data_fixture:
        datum = copy.deepcopy(datum)
        index.append(datum.pop('ophys_session_id'))
        data.append(datum)

    df = pd.DataFrame(
             data,
             index=pd.Index(index, name='ophys_session_id'))
    return df


@pytest.fixture(scope='session')
def ophys_experiment_data_fixture(ophys_session_data_fixture,
                                  experiment_state_lookup,
                                  container_state_lookup,
                                  ophys_experiment_to_container_map):
    """
    List of dicts.
    Each dict is an entry in the ophys_experiment_table as returned
    by the fetch_api.
    """
    rng = np.random.default_rng(182312)

    isi_id = 4000
    ophys_experiment_list = []
    for ophys_session in ophys_session_data_fixture:
        for i_experiment in ophys_session['ophys_experiment_id']:
            cntr_id_list = ophys_experiment_to_container_map[i_experiment]
            for container_id in cntr_id_list:
                datum = {
                    'ophys_session_id': ophys_session['ophys_session_id'],
                    'session_type': ophys_session['session_type'],
                    'behavior_session_id':
                    ophys_session['behavior_session_id'],
                    'ophys_container_id': container_id,
                    'container_workflow_state':
                    container_state_lookup[container_id],
                    'experiment_workflow_state':
                    experiment_state_lookup[i_experiment],
                    'session_name': ophys_session['session_name'],
                    'date_of_acquisition':
                    ophys_session['date_of_acquisition'],
                    'isi_experiment_id': isi_id,
                    'imaging_depth': rng.integers(50, 200),
                    'targeted_tructure': 'VISp',
                    'published_at': ophys_session['date_of_acquisition'],
                    'ophys_experiment_id': i_experiment}
                ophys_experiment_list.append(datum)
    return ophys_experiment_list


@pytest.fixture()
def ophys_experiments_table(ophys_experiment_data_fixture):
    """
    The ophys_experiments_table as returned by the fetch_api
    (a dataframe)
    """
    data = []
    index = []
    for datum in ophys_experiment_data_fixture:
        datum = copy.deepcopy(datum)
        index.append(datum.pop('ophys_experiment_id'))
        data.append(datum)

    df = pd.DataFrame(
              data,
              index=pd.Index(index, name='ophys_experiment_id'))
    return df


@pytest.fixture()
def intermediate_behavior_table(behavior_session_table,
                                mock_api):
    """
    A dataframe created by adding/transfrming columns in
    behavior_session_table. This table is used to produce the
    expected experiments_table and ophys_session_table.
    """
    df = behavior_session_table.copy(deep=True)

    df['reporter_line'] = df['reporter_line'].apply(
        ReporterLine.parse)
    df['cre_line'] = df['full_genotype'].apply(
        lambda x: FullGenotype(full_genotype=x).parse_cre_line())
    df['indicator'] = df['reporter_line'].apply(
        lambda x: ReporterLine(reporter_line=x).parse_indicator())

    df['prior_exposures_to_session_type'] = \
        get_prior_exposures_to_session_type(df=df)
    df['prior_exposures_to_image_set'] = \
        get_prior_exposures_to_image_set(df=df)
    df['prior_exposures_to_omissions'] = \
        get_prior_exposures_to_omissions(
            df=df,
            fetch_api=mock_api)
    return df


@pytest.fixture()
def expected_behavior_session_table(intermediate_behavior_table,
                                    ophys_session_data_fixture,
                                    mock_api,
                                    container_state_lookup,
                                    experiment_state_lookup,
                                    ophys_experiment_to_container_map,
                                    request):
    """
    The behavior_session_table as returned by the user-facing methods
    in behavior_project_cache.

    Note: request specifies whether the table was produced with
    passed_only = True or False. The actual object returned by
    this fixture is a dict. 'df' points to the dataframe.
    'passed_only' points to the value of passed_only used to
    generate the dataframe.
    """
    if hasattr(request, 'param'):
        passed_only = request.param
    else:
        passed_only = True

    df = intermediate_behavior_table.copy(deep=True)

    df['session_name_behavior'] = df['session_name']
    df = df.drop(['session_name'], axis=1)
    df['specimen_id_behavior'] = df['specimen_id']
    df = df.drop(['specimen_id'], axis=1)

    df['project_code'] = None
    df['ophys_session_id'] = None
    df['session_name_ophys'] = None
    df['ophys_experiment_id'] = None
    df['ophys_container_id'] = None
    df['specimen_id_ophys'] = None

    session_number = []
    for v in df['session_type'].values:
        if 'OPHYS' in v:
            session_number.append(1)
        else:
            session_number.append(None)
    df['session_number'] = session_number

    for ophys_session in ophys_session_data_fixture:
        index = ophys_session['behavior_session_id']
        df.at[index, 'project_code'] = ophys_session['project_code']
        df.at[index, 'ophys_session_id'] = ophys_session['ophys_session_id']
        df.at[index, 'session_name_ophys'] = ophys_session['session_name']

        container_id_list = set()
        exp_id_list = set()
        for exp_id in ophys_session['ophys_experiment_id']:
            # because SessionsTable does not filter on experiment state
            exp_id_list.add(exp_id)
            if experiment_state_lookup[exp_id] != 'passed' and passed_only:
                continue
            for container_id in ophys_experiment_to_container_map[exp_id]:
                is_published = (container_state_lookup[container_id]
                                == 'published')
                if is_published or not passed_only:
                    container_id_list.add(container_id)

        exp_id_list = list(exp_id_list)
        exp_id_list.sort()
        container_id_list = list(container_id_list)
        container_id_list.sort()

        df.at[index, 'ophys_container_id'] = container_id_list
        df.at[index, 'ophys_experiment_id'] = exp_id_list
        df.at[index, 'specimen_id_ophys'] = ophys_session['specimen_id']

    df['ophys_session_id'] = df['ophys_session_id'].astype(float)

    return {'df': df, 'passed_only': passed_only}


@pytest.fixture()
def expected_experiments_table(ophys_experiments_table,
                               container_state_lookup,
                               experiment_state_lookup,
                               intermediate_behavior_table,
                               request):
    """
    The experiments_table as returned by the user-facing methods
    in the behavior_project_cache

    Note: request specifies whether the table was produced with
    passed_only = True or False. The actual object returned by
    this fixture is a dict. 'df' points to the dataframe.
    'passed_only' points to the value of passed_only used to
    generate the dataframe.
    """

    if hasattr(request, 'param'):
        passed_only = request.param
    else:
        passed_only = True

    behavior_table = intermediate_behavior_table.copy(deep=True)
    expected = ophys_experiments_table.copy(deep=True)

    if passed_only:
        expected = expected.query("experiment_workflow_state=='passed'")
        expected = expected.query("container_workflow_state=='published'")

    expected = expected.join(behavior_table[
                                 ['equipment_name',
                                  'donor_id',
                                  'full_genotype',
                                  'mouse_id',
                                  'driver_line',
                                  'sex',
                                  'age_in_days',
                                  'foraging_id',
                                  'reporter_line',
                                  'specimen_id',
                                  'prior_exposures_to_session_type',
                                  'prior_exposures_to_image_set',
                                  'prior_exposures_to_omissions',
                                  'indicator',
                                  'cre_line']],
                             on='behavior_session_id')

    expected = expected.join(behavior_table[
                                 ['session_name']],
                             on='behavior_session_id',
                             rsuffix='_behavior')

    session_number = []
    for v in expected['session_type'].values:
        if 'OPHYS' in v:
            session_number.append(1)
        else:
            session_number.append(None)
    expected['session_number'] = session_number

    expected = add_experience_level_to_experiment_table(expected)
    expected = add_passive_flag_to_ophys_experiment_table(expected)
    expected = add_image_set_to_experiment_table(expected)

    expected['session_name_ophys'] = expected['session_name']
    expected = expected.drop(['session_name'], axis=1)

    return {'df': expected, 'passed_only': passed_only}


@pytest.fixture()
def expected_ophys_session_table(ophys_session_table,
                                 intermediate_behavior_table,
                                 container_state_lookup,
                                 experiment_state_lookup,
                                 ophys_experiment_to_container_map,
                                 request):
    """
    The ophys_session_table as returned by the user-facing methods
    in the behavior_project_cache.

    Note: request specifies whether the table was produced with
    passed_only = True or False. The actual object returned by
    this fixture is a dict. 'df' points to the dataframe.
    'passed_only' points to the value of passed_only used to
    generate the dataframe.
    """
    if hasattr(request, 'param'):
        passed_only = request.param
    else:
        passed_only = True
    expected = ophys_session_table.copy(deep=True)

    if passed_only:
        valid_containers = set()
        valid_experiments = set()
        for exp_id in ophys_experiment_to_container_map:
            if experiment_state_lookup[exp_id] != 'passed':
                continue
            for container_id in ophys_experiment_to_container_map[exp_id]:
                if container_state_lookup[container_id] == 'published':
                    valid_containers.add(container_id)
                    valid_experiments.add(exp_id)

        # ophys_sessions_table does not appear to filter on
        # whether or not an experiment is 'passed';
        # that is probably supposed to happen at the level
        # of the LIMS query (?)
        for index_val in expected.index.values:
            raw_containers = expected.loc[index_val]['ophys_container_id']
            container_id = [c for c in raw_containers if c in valid_containers]
            expected.at[index_val, 'ophys_container_id'] = container_id

    behavior_table = intermediate_behavior_table.copy(deep=True)

    expected = expected.join(behavior_table[
                                 ['equipment_name',
                                  'donor_id',
                                  'full_genotype',
                                  'mouse_id',
                                  'driver_line',
                                  'sex',
                                  'age_in_days',
                                  'foraging_id',
                                  'reporter_line',
                                  'prior_exposures_to_session_type',
                                  'prior_exposures_to_image_set',
                                  'prior_exposures_to_omissions',
                                  'indicator',
                                  'cre_line']],
                             on='behavior_session_id')

    expected = expected.join(
                  behavior_table[['specimen_id', 'session_name']],
                  on='behavior_session_id',
                  rsuffix='_behavior',
                  lsuffix='_ophys')

    session_number = []
    for v in expected['session_type'].values:
        if 'OPHYS' in v:
            session_number.append(1)
        else:
            session_number.append(None)
    expected['session_number'] = session_number

    return {'df': expected, 'passed_only': passed_only}


@pytest.fixture
def mock_api(ophys_session_table,
             behavior_session_table,
             ophys_experiments_table):

    class MockApi:

        def get_ophys_session_table(self):
            return ophys_session_table

        def get_behavior_session_table(self):
            return behavior_session_table

        def get_ophys_experiment_table(self):
            return ophys_experiments_table

        def get_session_data(self, ophys_session_id):
            return ophys_session_id

        def get_behavior_stage_parameters(self, foraging_ids):
            return {x: {} for x in foraging_ids}

    return MockApi


@pytest.fixture
def TempdirBehaviorCache(mock_api, request):
    temp_dir = tempfile.TemporaryDirectory()
    manifest = os.path.join(temp_dir.name, "manifest.json")
    yield VisualBehaviorOphysProjectCache(fetch_api=mock_api(),
                                          cache=request.param,
                                          manifest=manifest)
    temp_dir.cleanup()
