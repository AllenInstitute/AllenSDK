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


def from_json_cache_key(cls, dict_repr: dict, file_type: str):
    return hashkey(json.dumps(dict_repr))


class EcephysStimulusFile(DataFile):
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains a number of parameters collected during a behavior
    session including information about stimulus presentations, rewards,
    trials, and timing for all of the above.
    """

    def __init__(self, filepath: Union[str, Path], file_type: str):

        self._file_type = file_type

        super().__init__(filepath=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(cls, dict_repr: dict, file_type: str) -> "StimulusFile":
        filepath = dict_repr[file_type]
        return cls(filepath=filepath, file_type=file_type)

    def to_json(self) -> Dict[str, str]:
        return {"ecephye_behavior_stimulus_file": str(self.filepath)}

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> dict:
        filepath = safe_system_path(file_name=filepath)
        return pd.read_pickle(filepath)
