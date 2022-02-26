import json
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path

from allensdk.brain_observatory.ecephys.data_files \
    ._data_file_abc import \
    DataFile


def from_json_cache_key(cls, dict_repr: dict):
    return hashkey(json.dumps(dict_repr))


class EcephysStimulusTable(DataFile):
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains a number of parameters collected during a behavior
    session including information about stimulus presentations, rewards,
    trials, and timing for all of the above.
    """

    def __init__(self, filepath: Union[str, Path]):
        self._filepath = filepath

        # super().__init__(filepath=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(cls, dict_repr: dict) -> "EcephysStimulusTable":
        filepath = dict_repr['stim_table_file']
        return cls(filepath=filepath)

    def to_json(self) -> Dict[str, str]:
        return {"behavior_stimulus_table": str(self.filepath)}

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> dict:
        filepath = safe_system_path(file_name=filepath)
        return pd.read_pickle(filepath)
