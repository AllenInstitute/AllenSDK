import pytest
import pathlib
import copy
import json
import pandas as pd
import tempfile

from allensdk.brain_observatory.vbn_2022.metadata_writer \
    .metadata_writer import VBN2022MetadataWriterClass


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
        'on_missing_file, with_supplement',
        [('skip', False), ('warn', False), ('warn', True)])
def test_metadata_writer_smoketest(
        smoketest_config_fixture,
        tmp_path_factory,
        helper_functions,
        on_missing_file,
        with_supplement):
    """
    smoke test for VBN 2022 metadata writer. Requires LIMS
    and mtrain connections.

    If with_supplement is True, add supplemental data
    to the ecephys_sessions.csv file and test for their existence
    """

    output_names = ('units.csv', 'probes.csv', 'channels.csv',
                    'ecephys_sessions.csv', 'behavior_sessions.csv')

    config = copy.deepcopy(smoketest_config_fixture)

    output_dir = tmp_path_factory.mktemp('vbn_metadata_smoketest')
    output_dir = pathlib.Path(output_dir)
    output_json_path = pathlib.Path(
                           tempfile.mkstemp(dir=output_dir,
                                            suffix='.json')[1])
    config['output_dir'] = str(output_dir.resolve().absolute())

    if with_supplement:
        # add suplemental columns to configuration
        supplement = [{'ecephys_session_id': 1115077618,
                       'supplementA': 'cat',
                       'supplementB': 5},
                      {'ecephys_session_id': 1081429294,
                       'supplementA': 'frog',
                       'supplementB': 6},
                      {'ecephys_session_id': 11111,
                       'supplementA': None,
                       'supplementB': None}]

        config['supplemental_data'] = supplement

    expected_paths = []
    for name in output_names:
        file_path = output_dir / name
        assert not file_path.exists()
        expected_paths.append(file_path)

    this_dir = pathlib.Path('.')
    this_dir = str(this_dir.resolve().absolute())

    config['ecephys_nwb_prefix'] = 'not_there'
    config['ecephys_nwb_dir'] = this_dir
    config['clobber'] = False
    config['on_missing_file'] = on_missing_file
    config['output_json'] = str(output_json_path.resolve().absolute())

    writer = VBN2022MetadataWriterClass(args=[], input_data=config)
    writer.run()

    # load a dict mapping the name of a metadata.csv
    # to the list of columns it is supposed to contain
    this_dir = pathlib.Path(__file__).parent
    resource_dir = this_dir / 'resources'
    with open(resource_dir / 'column_lookup.json', 'rb') as in_file:
        column_lookup = json.load(in_file)

    for file_path in expected_paths:
        assert file_path.exists()
        df = pd.read_csv(file_path)
        expected_columns = set(column_lookup[file_path.name])
        if with_supplement and file_path.name == 'ecephys_sessions.csv':
            expected_columns.add('supplementA')
            expected_columns.add('supplementB')

        actual_columns = set(df.columns)
        assert expected_columns == actual_columns

    if with_supplement:
        df = pd.read_csv(output_dir / 'ecephys_sessions.csv')
        # make sure that no extra rows were added when adding
        # the supplemental data
        assert len(df) == 2

        for expected in supplement[:2]:
            this_row = df.loc[
                df.ecephys_session_id == expected['ecephys_session_id']]
            assert len(this_row) == 1
            assert this_row.supplementA.values[0] == expected['supplementA']
            assert this_row.supplementB.values[0] == expected['supplementB']

    helper_functions.windows_safe_cleanup_dir(
        dir_path=output_dir)


@pytest.mark.requires_bamboo
def test_with_failed_sessions(
        smoketest_with_failed_sessions_config_fixture,
        tmp_path_factory,
        helper_functions):
    """
    Test that metadata writer CLI can handle
    failed_ecephys_session_id_list
    """
    output_dir = pathlib.Path(tmp_path_factory.mktemp('failed_session_test'))
    output_json_path = output_dir / 'output.json'

    config = copy.deepcopy(smoketest_with_failed_sessions_config_fixture)
    config['clobber'] = False
    config['output_dir'] = str(output_dir.resolve().absolute())
    config['on_missing_file'] = 'warn'
    config['ecephys_nwb_prefix'] = 'not_here'
    config['ecephys_nwb_dir'] = str(output_dir.resolve().absolute())
    config['output_json'] = str(output_json_path.resolve().absolute())

    writer = VBN2022MetadataWriterClass(args=[], input_data=config)
    writer.run()

    for fname in ('behavior_sessions.csv',
                  'ecephys_sessions.csv',
                  'channels.csv',
                  'units.csv',
                  'probes.csv'):
        file_path = output_dir / fname
        assert file_path.is_file()

    ecephys_sessions_df = pd.read_csv(output_dir / 'ecephys_sessions.csv')

    assert len(ecephys_sessions_df) == 1

    for bad_session_id in config['failed_ecephys_session_id_list']:
        assert (bad_session_id not in
                ecephys_sessions_df.ecephys_session_id.values)

    for good_session_id in config['ecephys_session_id_list']:
        assert (good_session_id in
                ecephys_sessions_df.ecephys_session_id.values)

    # make sure this session was recorded with session_number = 2
    assert ecephys_sessions_df.session_number.values[0] == 2

    helper_functions.windows_safe_cleanup_dir(
            dir_path=output_dir)
