import os

import pandas as pd
import numpy as np


def structure_assignment_fname_map():
    return {
        733744647: "732592105_probeB_733744647.csv",
        733744649: "732592105_probeC_733744649.csv",
        733744651: "732592105_probeD_733744651.csv",
        733744653: "732592105_probeE_733744653.csv",
        733744655: "732592105_probeF_733744655.csv",
        805579698: "799864342_probeA_805579698.csv",
        832129149: "829720705_probeA_832129149.csv",
        837761710: "835479236_probeC_837761710.csv"
    }


def load_structure_assignments(probe_ids=None):

    fnames = structure_assignment_fname_map()
    dirname = os.path.join(os.path.dirname(__file__), "data")

    output = []
    for probe_id, fname in fnames.items():
        if probe_ids is not None and probe_id not in probe_ids:
            continue

        full_path = os.path.join(dirname, fname)
        df = pd.read_csv(full_path)

        df["ecephys_probe_id"] = np.zeros([len(df)], dtype=int) + probe_id
        output.append(df)

    return pd.concat(output)


def replace_bad_structure_assignments(
    channels, 
    structure_id_key="ecephys_structure_id", 
    structure_acronym_key="ecephys_structure_acronym",
    inplace=False
):
    if not inplace:
        channels = channels.copy()

    probes = set(channels["ecephys_probe_id"].unique().tolist())

    assignments = load_structure_assignments()
    reassigned_probes = set(assignments["ecephys_probe_id"].unique().tolist())

    reassigned = channels[channels["ecephys_probe_id"].isin(reassigned_probes)]
    reassigned = pd.merge(
        reassigned.reset_index(), assignments,
        how="left",
        left_on=["ecephys_probe_id", "local_index"], 
        right_on=["ecephys_probe_id", "local_index"], 
        suffixes=["_original", "_corrected"]
    )
    reassigned.set_index("id", inplace=True)

    reassigned.drop(columns=[
        f"{structure_id_key}_original", 
        f"{structure_acronym_key}_original"
    ], inplace=True)
    reassigned.rename(columns={
        f"{structure_id_key}_corrected": structure_id_key, 
        f"{structure_acronym_key}_corrected": structure_acronym_key
    }, inplace=True)

    channels.loc[reassigned.index.values, [structure_id_key, structure_acronym_key]] = \
        reassigned.loc[:, [structure_id_key, structure_acronym_key]]

    if not inplace:
        return reassigned