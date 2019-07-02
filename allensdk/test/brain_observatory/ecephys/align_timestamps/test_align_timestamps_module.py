import json
import os
import subprocess as sp

import pytest
import numpy as np


DATA_DIR = os.environ.get(
    "ECEPHYS_PIPELINE_DATA",
    os.path.join("/", "allen", "aibs", "informatics", "module_test_data", "ecephys"),
)


def apply_input_json_template(
    template_path, input_json_path, temp_dir, data_dir=DATA_DIR
):
    """ A utility for adjusting the input json so that:
        1. input paths find cached data in the data dir
        2. output paths write to a specified temp_dir
    The adjusted input json will be written to temp_dir.

    """

    with open(template_path, "r") as input_json_file:
        input_json_data = json.load(input_json_file)

    input_json_data["sync_h5_path"] = os.path.join(
        data_dir, input_json_data["sync_h5_path"]
    )

    for probe in input_json_data["probes"]:

        probe["barcode_channel_states_path"] = os.path.join(
            data_dir, probe["barcode_channel_states_path"]
        )
        probe["barcode_timestamps_path"] = os.path.join(
            data_dir, probe["barcode_timestamps_path"]
        )

        for timestamps_file in probe["mappable_timestamp_files"]:
            timestamps_file["input_path"] = os.path.join(
                data_dir, timestamps_file["input_path"]
            )
            timestamps_file["output_path"] = os.path.join(
                temp_dir, timestamps_file["output_path"]
            )

    with open(input_json_path, "w") as input_json_file:
        json.dump(input_json_data, input_json_file)


@pytest.fixture()
def align_timestamps_706875901_expected_params():
    return {
        "probeA": {
            "total_time_shift": -0.6097051644554128,
            "global_probe_sampling_rate": 29999.956819421783,
            "global_probe_lfp_sampling_rate": 2499.9964016184817,
        },
        "probeB": {
            "total_time_shift": -0.5875482733055364,
            "global_probe_sampling_rate": 29999.90905329544,
            "global_probe_lfp_sampling_rate": 2499.9924211079533,
        },
    }


@pytest.fixture()
def align_timestamps_706875901_expected_files():
    return lambda data_dir: {
        "probeA": {
            "spikes_timestamps": os.path.join(
                data_dir, "706875901_probeA_aligned_spike_timestamps.npy"
            ),
            "lfp_timestamps": os.path.join(
                data_dir, "706875901_probeA_aligned_lfp_timestamps.npy"
            ),
        },
        "probeB": {
            "spikes_timestamps": os.path.join(
                data_dir, "706875901_probeB_aligned_spike_timestamps.npy"
            ),
            "lfp_timestamps": os.path.join(
                data_dir, "706875901_probeB_aligned_lfp_timestamps.npy"
            ),
        },
    }


@pytest.fixture(scope="module")
def run_align_timestamps_706875901(tmpdir_factory):
    base_path = tmpdir_factory.mktemp("align_timestamps_integration")
    executable = ["python", "-m", "allensdk.brain_observatory.ecephys.align_timestamps"]

    input_json_path = os.path.join(base_path, "706875901_align_timestamps_input.json")
    output_json_path = os.path.join(base_path, "706875901_align_timestamps_output.json")
    executable.extend(["--input_json", input_json_path])
    executable.extend(["--output_json", output_json_path])

    input_json_template_path = os.path.join(
        DATA_DIR, "706875901_align_timestamps_input.json"
    )
    apply_input_json_template(input_json_template_path, input_json_path, base_path)

    sp.check_call(executable)

    return output_json_path


@pytest.mark.requires_bamboo
def test_align_timestamps_parameters_706875901(
    run_align_timestamps_706875901, align_timestamps_706875901_expected_params
):

    with open(run_align_timestamps_706875901, "r") as output_json_file:
        output_json_data = json.load(output_json_file)

    for probe in output_json_data["probe_outputs"]:
        expected = align_timestamps_706875901_expected_params[probe["name"]]

        assert expected["total_time_shift"] == probe["total_time_shift"]
        assert (
            expected["global_probe_sampling_rate"]
            == probe["global_probe_sampling_rate"]
        )
        assert (
            expected["global_probe_lfp_sampling_rate"]
            == probe["global_probe_lfp_sampling_rate"]
        )


@pytest.mark.requires_bamboo
def test_align_timestamps_files_706875901(
    run_align_timestamps_706875901, align_timestamps_706875901_expected_files
):

    with open(run_align_timestamps_706875901, "r") as output_json_file:
        output_json_data = json.load(output_json_file)

    expected_files = align_timestamps_706875901_expected_files(DATA_DIR)
    for probe in output_json_data["probe_outputs"]:

        for output_file_key, output_file_path in probe["output_paths"].items():
            expected_file_path = expected_files[probe["name"]][output_file_key]
            expected_data = np.load(expected_file_path, allow_pickle=False)

            obtained_data = np.load(output_file_path, allow_pickle=False)

            assert np.allclose(expected_data, obtained_data)


@pytest.mark.requires_bamboo
def test_align_timestamps_barcode_agreement_706875901(run_align_timestamps_706875901):

    with open(run_align_timestamps_706875901, "r") as output_json_file:
        output_json_data = json.load(output_json_file)

    probe_parameters = {}
    for probe in output_json_data["probe_outputs"]:
        probe_parameters[probe["name"]] = probe

    aligned_barcode_data = []
    barcode_timestamp_lengths = []
    for probe in output_json_data["input_parameters"]["probes"]:
        name = probe["name"]
        barcode_data = np.load(probe["barcode_timestamps_path"], allow_pickle=False)

        total_time_shift = probe_parameters[name]["total_time_shift"]
        global_probe_sampling_rate = probe_parameters[name][
            "global_probe_sampling_rate"
        ]

        aligned_barcode_data.append(
            barcode_data / global_probe_sampling_rate - total_time_shift
        )
        barcode_timestamp_lengths.append(len(aligned_barcode_data))

    min_length = np.amin(barcode_timestamp_lengths)
    assert min_length > 0

    for ii in range(len(aligned_barcode_data) - 1):
        assert np.allclose(
            aligned_barcode_data[ii][:min_length],
            aligned_barcode_data[ii + 1][:min_length],
        )
