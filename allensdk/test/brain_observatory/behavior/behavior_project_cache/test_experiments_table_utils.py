import copy
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    tables.util.experiments_table_utils import (
        add_experience_level_to_experiment_table,
        add_passive_flag_to_ophys_experiment_table,
        add_image_set_to_experiment_table)


def test_add_experience_level():

    input_data = []
    expected_data = []

    datum = {'id': 0,
             'session_number': 1,
             'prior_exposures_to_image_set': 4}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 1,
             'session_number': 2,
             'prior_exposures_to_image_set': 5}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 2,
             'session_number': 3,
             'prior_exposures_to_image_set': 1772}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 3,
             'session_number': 4,
             'prior_exposures_to_image_set': 0}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel 1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 4,
             'session_number': 5,
             'prior_exposures_to_image_set': 0}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'None'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 5,
             'session_number': 6,
             'prior_exposures_to_image_set': 0}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'None'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 7,
             'session_number': 4,
             'prior_exposures_to_image_set': 2}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 8,
             'session_number': 5,
             'prior_exposures_to_image_set': 1}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 9,
             'session_number': 6,
             'prior_exposures_to_image_set': 3}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 10,
             'session_number': 7,
             'prior_exposures_to_image_set': 3}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'None'
    expected_data.append(copy.deepcopy(datum))

    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)
    output_df = add_experience_level_to_experiment_table(input_df)
    assert not input_df.equals(output_df)
    assert len(input_df.columns) != len(output_df.columns)
    assert output_df.equals(expected_df)


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
