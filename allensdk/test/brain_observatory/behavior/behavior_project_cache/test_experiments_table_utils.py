import copy
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    tables.util.experiments_table_utils import (
        add_passive_flag_to_ophys_experiment_table,
        add_image_set_to_experiment_table)


def test_add_passive_flag():

    input_data = []
    expected_data = []

    datum = {'id': 0, 'session_number': 2}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = True
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 1, 'session_number': 5}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = True
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 2, 'session_number': 1}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = False
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 3, 'session_number': 3}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = False
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 4, 'session_number': 2}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = True
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 5, 'session_number': 5}
    input_data.append(copy.deepcopy(datum))
    datum['passive'] = True
    expected_data.append(copy.deepcopy(datum))

    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)
    assert not input_df.equals(expected_df)
    output_df = add_passive_flag_to_ophys_experiment_table(
                    input_df)
    assert not input_df.equals(output_df)
    assert len(input_df.columns) != len(output_df.columns)
    assert output_df.equals(expected_df)


def test_add_image_set_to_experiment_table():

    input_data = []
    expected_data = []

    datum = {'id': 0, 'session_type': 'ophys_5_images_x_passive'}
    input_data.append(copy.deepcopy(datum))
    datum['image_set'] = 'x'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 1, 'session_type': 'ophys_5'}
    input_data.append(copy.deepcopy(datum))
    datum['image_set'] = 'N/A'
    expected_data.append(copy.deepcopy(datum))

    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)
    assert not expected_df.equals(input_df)
    output_df = add_image_set_to_experiment_table(input_df)
    assert not input_df.equals(output_df)
    assert len(input_df.columns) != len(output_df.columns)
    assert output_df.equals(expected_df)
