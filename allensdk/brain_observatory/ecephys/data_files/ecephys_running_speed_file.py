import json
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import h5py as h5

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.brain_observatory.behavior.data_files import DataFile

# Query returns path to RunningSpeed file for given behavior session
def from_json_cache_key(cls, dict_repr: dict):
    return hashkey(json.dumps(dict_repr))


def from_lims_cache_key(cls, db, behavior_session_id: int):
    return hashkey(behavior_session_id)


class EcephysRunningSpeedFile:
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains a number of parameters collected during a behavior
    session including information about stimulus presentations, rewards,
    trials, and timing for all of the above.
    """

    def __init__(self, filepath: Union[str, Path]):
        # super().__init__(filepath=filepath)
        self._filepath = filepath
        self.data = self.load()

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(cls, dict_repr: dict) -> "EcephysRunningSpeedFile":
        filepath = dict_repr["running_speed_path"]
        return cls(filepath=filepath)


    def load(self):
        """
        Loads an hdf5 sync dataset.

        Parameters
        ----------
        path : str
            Path to hdf5 file.

        """

        return h5.File(self._filepath, 'r')

    @staticmethod
    def from_lims():
        pass

    # @staticmethod
    def to_json():
        pass
