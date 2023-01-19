import copy
import tempfile
from pathlib import Path

import pytest
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession,
)
from allensdk.brain_observatory.nwb import nwb_utils
from allensdk.brain_observatory.nwb.nwb_utils import NWBWriter


@pytest.mark.parametrize(
    "input_cols, possible_names, expected_intersection",
    [
        (
            ["duration", "end_frame", "image_index", "image_name"],
            {"stimulus_name", "image_name"},
            "image_name",
        ),
        (
            ["duration", "end_frame", "image_index", "stimulus_name"],
            {"stimulus_name", "image_name"},
            "stimulus_name",
        ),
    ],
)
def test_get_stimulus_name_column(
    input_cols, possible_names, expected_intersection
):
    column_name = nwb_utils.get_column_name(input_cols, possible_names)
    assert column_name == expected_intersection


@pytest.mark.parametrize(
    "input_cols, possible_names, expected_excep_cols",
    [
        (
            ["duration", "end_frame", "image_index"],
            {"stimulus_name", "image_name"},
            [],
        ),
        (
            [
                "duration",
                "end_frame",
                "image_index",
                "image_name",
                "stimulus_name",
            ],
            {"stimulus_name", "image_name"},
            ["stimulus_name", "image_name"],
        ),
    ],
)
def test_get_stimulus_name_column_exceptions(
    input_cols, possible_names, expected_excep_cols
):
    with pytest.raises(KeyError) as error:
        nwb_utils.get_column_name(input_cols, possible_names)
    for expected_value in expected_excep_cols:
        assert expected_value in str(error.value)


@pytest.mark.requires_bamboo
def test_nwb_writer(behavior_ecephys_session_config_fixture):
    session_data = copy.deepcopy(behavior_ecephys_session_config_fixture)

    with tempfile.NamedTemporaryFile() as f:
        nwb_writer = NWBWriter(
            nwb_filepath=f.name,
            session_data=session_data,
            serializer=BehaviorSession,
        )
        # Skip the metadata comparison. We will check that in another test.
        nwb_writer.write_nwb(
            skip_metadata=list(session_data.keys()),
            read_stimulus_presentations_table_from_file=True,
            sync_file_permissive=True,
            running_speed_load_from_multiple_stimulus_files=True,
            include_experiment_description=False,
            stimulus_presentations_stimulus_column_name="stimulus_name",
            add_is_change_to_stimulus_presentations_table=False,
        )
        assert Path(f.name).exists()


@pytest.mark.requires_bamboo
def test_nwb_writer_compare_success(behavior_ecephys_session_config_fixture):
    session_data = copy.deepcopy(behavior_ecephys_session_config_fixture)

    with tempfile.NamedTemporaryFile() as f:
        nwb_writer = NWBWriter(
            nwb_filepath=f.name,
            session_data=session_data,
            serializer=BehaviorSession,
        )
        skip_metadata = copy.deepcopy(session_data)
        skip_metadata.pop("mouse_id", None)
        nwb_writer.write_nwb(
            skip_metadata=list(skip_metadata.keys()),
            read_stimulus_presentations_table_from_file=True,
            sync_file_permissive=True,
            running_speed_load_from_multiple_stimulus_files=True,
            include_experiment_description=False,
            stimulus_presentations_stimulus_column_name="stimulus_name",
            add_is_change_to_stimulus_presentations_table=False,
        )
        assert Path(f.name).exists()


@pytest.mark.requires_bamboo
def test_nwb_writer_compare_failure(behavior_ecephys_session_config_fixture):
    session_data = copy.deepcopy(behavior_ecephys_session_config_fixture)

    with pytest.raises(FileNotFoundError):
        with tempfile.NamedTemporaryFile() as f:
            nwb_writer = NWBWriter(
                nwb_filepath=f.name,
                session_data=session_data,
                serializer=BehaviorSession,
            )
            skip_metadata = copy.deepcopy(session_data)
            skip_metadata.pop("mouse_id", None)
            session_data["mouse_id"] = "1234"
            with pytest.raises(ValueError, match="Value for mouse_id"):
                nwb_writer.write_nwb(
                    skip_metadata=list(skip_metadata.keys()),
                    read_stimulus_presentations_table_from_file=True,
                    sync_file_permissive=True,
                    running_speed_load_from_multiple_stimulus_files=True,
                    include_experiment_description=False,
                    stimulus_presentations_stimulus_column_name="stimulus_name",  # noqa: E501
                    add_is_change_to_stimulus_presentations_table=False,
                )
