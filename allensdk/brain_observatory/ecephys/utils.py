import logging

import numpy as np
import pandas as pd


def load_and_squeeze_npy(path):
    return np.squeeze(np.load(path, allow_pickle=False))


def group_1d_by_unit(data, data_unit_map, local_to_global_unit_map=None):
    sort_order = np.argsort(data_unit_map, kind="stable")
    data_unit_map = data_unit_map[sort_order]
    data = data[sort_order]

    changes = np.concatenate(
        [
            np.array([0]),
            np.where(np.diff(data_unit_map))[0] + 1,
            np.array([data.size]),
        ]
    )

    output = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = data_unit_map[low]
        current = data[low:high]

        if local_to_global_unit_map is not None:
            if local_unit not in local_to_global_unit_map:
                logging.warning(
                    f"unable to find unit at local position {local_unit}"
                )
                continue
            global_id = local_to_global_unit_map[local_unit]
            output[global_id] = current
        else:
            output[local_unit] = current

    return output


def scale_amplitudes(spike_amplitudes,
                     templates,
                     spike_templates,
                     scale_factor=1.0):

    template_full_amplitudes = templates.max(axis=1) - templates.min(axis=1)
    template_amplitudes = template_full_amplitudes.max(axis=1)

    template_amplitudes = template_amplitudes[spike_templates]
    spike_amplitudes = template_amplitudes * spike_amplitudes * scale_factor
    return spike_amplitudes


def clobbering_merge(to_df, from_df, **kwargs):
    overlapping = set(to_df.columns) & set(from_df.columns)

    for merge_param in ["on", "left_on", "right_on"]:
        if merge_param in kwargs:
            merge_arg = kwargs.get(merge_param)
            if isinstance(merge_arg, str):
                merge_arg = [merge_arg]
            overlapping = overlapping - set(list(merge_arg))

    to_df = to_df.drop(columns=list(overlapping))
    return pd.merge(to_df, from_df, **kwargs)
