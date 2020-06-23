import pytest
import pandas as pd

from allensdk.brain_observatory.nwb import nwb_utils


@pytest.mark.parametrize("input_cols, possible_names, expected_intersection", [
    (['duration', 'end_frame', 'image_index', 'image_name'],
     {'stimulus_name', 'image_name'}, 'image_name'),
    (['duration', 'end_frame', 'image_index', 'stimulus_name'],
     {'stimulus_name', 'image_name'}, 'stimulus_name')
])
def test_get_stimulus_name_column(input_cols, possible_names,
                                  expected_intersection):
    column_name = nwb_utils.get_column_name(input_cols, possible_names)
    assert column_name == expected_intersection


@pytest.mark.parametrize("input_cols, possible_names, expected_excep_cols", [
    (['duration', 'end_frame', 'image_index'], {'stimulus_name', 'image_name'},
     []),
    (['duration', 'end_frame', 'image_index', 'image_name', 'stimulus_name'],
     {'stimulus_name', 'image_name'},
     ['stimulus_name', 'image_name'])
])
def test_get_stimulus_name_column_exceptions(input_cols,
                                             possible_names,
                                             expected_excep_cols):
    with pytest.raises(KeyError) as error:
        nwb_utils.get_column_name(input_cols, possible_names)
    for expected_value in expected_excep_cols:
        assert expected_value in str(error.value)


@pytest.mark.parametrize("stimulus_table, expected_table_data", [
    ({'image_index': [8, 9],
      'image_name': ['omitted', 'not_omitted'],
      'image_set': ['omitted', 'not_omitted'],
      'index': [201, 202],
      'omitted': [True, False],
      'start_frame': [231060, 232340],
      'start_time': [0, 250],
      'stop_time': [None, 1340509]},
     {'image_index': [8, 9],
      'image_name': ['omitted', 'not_omitted'],
      'image_set': ['omitted', 'not_omitted'],
      'index': [201, 202],
      'omitted': [True, False],
      'start_frame': [231060, 232340],
      'start_time': [0, 250],
      'stop_time': [0.25, 1340509]}
     )
])
def test_set_omitted_stop_time(stimulus_table, expected_table_data):
    stimulus_table = pd.DataFrame.from_dict(data=stimulus_table)
    expected_table = pd.DataFrame.from_dict(data=expected_table_data)
    nwb_utils.set_omitted_stop_time(stimulus_table)
    assert stimulus_table.equals(expected_table)
