import pytest
import json
import pathlib

from allensdk.brain_observatory.vbn_2022.input_json_writer \
    .input_json_writer import VBN2022InputJsonWriter

from allensdk.brain_observatory.ecephys.write_nwb.vbn._schemas import (
    VBNInputSchema)


@pytest.mark.requires_bamboo
def test_writer_cli(
        tmp_path_factory,
        helper_functions):
    """
    This will just be a smoke test to write out the input json
    and validate it against the NWB writer schema
    """

    base_dir = pathlib.Path(tmp_path_factory.mktemp('input_json_cli'))

    json_dir = base_dir / 'input_jsons'
    json_dir.mkdir()

    nwb_dir = base_dir / 'nwb_files'
    nwb_dir.mkdir()

    json_prefix = "test_input_jsons"
    nwb_prefix = "test_vbn_nwb"

    # ecephys_session_id = 9 is meant to provide an obvious
    # non-existent session and make sure the code fails
    # gracefully

    session_list = [1117148442, 1077897245, 9]

    probes_to_skip = []
    for session_id in session_list:
        for suffix in 'BDF':
            probes_to_skip.append({
                "session": session_id,
                "probe": f"probe{suffix}"
            })

    json_generation_data = {
        "log_level": "INFO",
        "ecephys_session_id_list": session_list,
        "clobber": False,
        "json_output_dir": str(json_dir.resolve().absolute()),
        "nwb_output_dir": str(nwb_dir.resolve().absolute()),
        "json_prefix": json_prefix,
        "nwb_prefix": nwb_prefix,
        "probes_to_skip": probes_to_skip}

    writer = VBN2022InputJsonWriter(args=[],
                                    input_data=json_generation_data)
    writer.run()

    ct_valid = 0
    for session_id in session_list:
        expected_path = json_dir / f"{json_prefix}_{session_id}_input.json"
        if session_id == 9:
            assert not expected_path.exists()
        else:

            ct_valid += 1
            assert expected_path.is_file()

            # read in the written file and verify that it
            # passes schema validation
            with open(expected_path, 'rb') as in_file:
                json_data = json.load(in_file)
            session_data = json_data['session_data']
            assert len(session_data['probes']) > 0
            assert len(session_data['probes'][0]['channels']) > 0
            assert len(session_data['probes'][0]['units']) > 0

            schema = VBNInputSchema()
            assert len(schema.validate(data=json_data)) == 0

    # make sure we actually tested some valid files
    assert ct_valid > 0

    helper_functions.windows_safe_cleanup_dir(
            dir_path=base_dir)
