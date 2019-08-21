import os
import sys
import copy

import numpy as np
from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
import argparse

from ._schemas import InputParameters, OutputParameters
from .barcode_sync_dataset import BarcodeSyncDataset
from .probe_synchronizer import ProbeSynchronizer
from . import barcode
from .channel_states import extract_barcodes_from_states, extract_splits_from_states


def align_timestamps(args):
    
    sync_dataset = BarcodeSyncDataset.factory(args["sync_h5_path"])
    sync_times, sync_codes = sync_dataset.extract_barcodes()

    probe_output_info = []
    for probe in args["probes"]:
        print(probe["name"])
        this_probe_output_info = {}

        channel_states = np.load(probe["barcode_channel_states_path"])
        timestamps = np.load(probe["barcode_timestamps_path"])

        probe_barcode_times, probe_barcodes = extract_barcodes_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )
        probe_split_times = extract_splits_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )

        print("Split times:")
        print(probe_split_times)

        synchronizers = []

        for idx, split_time in enumerate(probe_split_times):

            min_time = probe_split_times[idx]

            if idx == (len(probe_split_times) - 1):
                max_time = np.Inf
            else:
                max_time = probe_split_times[idx + 1]

            synchronizer = ProbeSynchronizer.compute(
                sync_times,
                sync_codes,
                probe_barcode_times,
                probe_barcodes,
                min_time,
                max_time,
                probe["start_index"],
                probe["sampling_rate"],
            )

            synchronizers.append(synchronizer)

        mapped_files = {}

        for timestamp_file in probe["mappable_timestamp_files"]:
            # print(timestamp_file["name"])
            timestamps = np.load(timestamp_file["input_path"])
            aligned_timestamps = np.copy(timestamps).astype("float64")

            for synchronizer in synchronizers:
                aligned_timestamps = synchronizer(aligned_timestamps)
                print("total time shift: " + str(synchronizer.total_time_shift))
                print(
                    "actual sampling rate: "
                    + str(synchronizer.global_probe_sampling_rate)
                )

            np.save(
                timestamp_file["output_path"], aligned_timestamps, allow_pickle=False
            )
            mapped_files[timestamp_file["name"]] = timestamp_file["output_path"]

        lfp_sampling_rate = (
            probe["lfp_sampling_rate"] * synchronizer.sampling_rate_scale
        )

        this_probe_output_info["total_time_shift"] = synchronizer.total_time_shift
        this_probe_output_info[
            "global_probe_sampling_rate"
        ] = synchronizer.global_probe_sampling_rate
        this_probe_output_info["global_probe_lfp_sampling_rate"] = lfp_sampling_rate
        this_probe_output_info["output_paths"] = mapped_files
        this_probe_output_info["name"] = probe["name"]

        probe_output_info.append(this_probe_output_info)

    return {"probe_outputs": probe_output_info}


def main():

    mod = ArgSchemaParserPlus(
        schema_type=InputParameters, output_schema_type=OutputParameters
    )
    output = align_timestamps(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()
