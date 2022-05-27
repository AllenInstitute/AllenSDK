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
        'on_missing_file', ['skip', 'warn'])
def test_metadata_writer_smoketest(
        smoketest_config_fixture,
        tmp_path_factory,
        helper_functions,
        on_missing_file):
    """
    smoke test for VBN 2022 metadata writer. Requires LIMS
    and mtrain connections.
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
        actual_columns = set(df.columns)
        assert expected_columns == actual_columns

    helper_functions.windows_safe_cleanup_dir(
        dir_path=output_dir)
