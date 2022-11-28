import json
from typing import Dict, Union
from pathlib import Path

import h5py
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core import DataFile


def from_lims_cache_key(cls, db, ophys_experiment_id: int):
    return hashkey(ophys_experiment_id)


class NeuropilFile(DataFile):
    """A DataFile which contains methods for accessing and loading
    neuropil traces.
    """

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin, ophys_experiment_id: Union[int, str]
    ) -> "NeuropilFile":
        query = """
                SELECT wkf.storage_directory || wkf.filename AS neuropil_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkf.attachable_type = 'OphysExperiment'
                AND wkft.name = 'OphysNeuropilTraces'
                AND oe.id = {};
                """.format(
            ophys_experiment_id
        )
        filepath = db.fetchone(query, strict=True)
        return cls(filepath=filepath)

    @staticmethod
    def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        with h5py.File(filepath, "r") as in_file:
            traces = in_file["data"][()]
            roi_id = in_file["roi_names"][()]
            idx = pd.Index(roi_id, name="cell_roi_id").astype("int64")
            return pd.DataFrame({"neuropil_trace": list(traces)}, index=idx)
