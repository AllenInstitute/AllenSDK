import pytest

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


@pytest.mark.parametrize("stimulus_row, expected_stop_time", [
    ({'image_index': 8,
      'image_name': 'omitted',
      'image_set': 'omitted',
      'index': 201,
      'omitted': True,
      'start_frame': 231060,
      'start_time': 0}, 0.250)
])
def test_set_omitted_stop_time(stimulus_row, expected_stop_time):
    nwb_utils.set_omitted_stop_time(stimulus_row)
    assert stimulus_row['stop_time'] == expected_stop_time


@pytest.mark.parametrize("stimulus_row", [
    ({'image_index': 8,
      'image_name': 'omitted',
      'image_set': 'omitted',
      'index': 201,
      'omitted': False,
      'start_frame': 231060,
      'start_time': 0}),
    ({'image_index': 8,
      'image_name': 'omitted',
      'image_set': 'omitted',
      'index': 201,
      'start_frame': 231060,
      'start_time': 0})
])
def test_set_omitted_stop_time_exceptions(stimulus_row):
    regex = f".* \\{stimulus_row}"
    with pytest.raises(ValueError, match=regex):
        nwb_utils.set_omitted_stop_time(stimulus_row)
