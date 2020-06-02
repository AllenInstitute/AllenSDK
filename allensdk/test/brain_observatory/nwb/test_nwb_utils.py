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
    column_name = nwb_utils.get_stimulus_name_column(input_cols, possible_names)
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
    regex = ".* \\" + str(expected_excep_cols)
    with pytest.raises(KeyError, match=regex):
        nwb_utils.get_stimulus_name_column(input_cols, possible_names)
