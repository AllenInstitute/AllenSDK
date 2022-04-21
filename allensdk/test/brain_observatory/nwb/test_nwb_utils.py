import json
import tempfile
from pathlib import Path

import pytest

from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from allensdk.brain_observatory.nwb import nwb_utils
from allensdk.brain_observatory.nwb.nwb_utils import NWBWriter


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


class TestNwbWriter:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1111216934_input.json') \
                as f:
            input_data = json.load(f)
        cls.session_data = input_data['session_data']

    @pytest.mark.requires_bamboo
    def test_nwb_writer(self):
        with tempfile.NamedTemporaryFile() as f:
            nwb_writer = NWBWriter(
                nwb_filepath=f.name,
                session_data=self.session_data,
                serializer=BehaviorSession
            )
            nwb_writer.write_nwb(
                read_stimulus_presentations_table_from_file=True,
                sync_file_permissive=True,
                running_speed_load_from_multiple_stimulus_files=True,
                include_experiment_description=False,
                stimulus_presentations_stimulus_column_name='stimulus_name',
                add_is_change_to_stimulus_presentations_table=False
            )
            assert Path(f.name).exists()
