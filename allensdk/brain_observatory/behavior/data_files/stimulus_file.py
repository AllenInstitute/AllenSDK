import json
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.brain_observatory.behavior.data_files import DataFile

# Query returns path to StimulusPickle file for given behavior session
STIMULUS_FILE_QUERY_TEMPLATE = """
    SELECT
        wkf.storage_directory || wkf.filename AS stim_file
    FROM
        well_known_files wkf
    WHERE
        wkf.attachable_id = {behavior_session_id}
        AND wkf.attachable_type = 'BehaviorSession'
        AND wkf.well_known_file_type_id IN (
            SELECT id
            FROM well_known_file_types
            WHERE name = 'StimulusPickle');
"""


def from_json_cache_key(cls, dict_repr: dict):
    return hashkey(json.dumps(dict_repr))


def from_lims_cache_key(cls, db, behavior_session_id: int):
    return hashkey(behavior_session_id)


class StimulusFile(DataFile):
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains a number of parameters collected during a behavior
    session including information about stimulus presentations, rewards,
    trials, and timing for all of the above.
    """

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(cls, dict_repr: dict) -> "StimulusFile":
        filepath = dict_repr["behavior_stimulus_file"]
        return cls(filepath=filepath)

    def to_json(self) -> Dict[str, str]:
        return {"behavior_stimulus_file": str(self.filepath)}

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        behavior_session_id: Union[int, str]
    ) -> "StimulusFile":
        query = STIMULUS_FILE_QUERY_TEMPLATE.format(
            behavior_session_id=behavior_session_id
        )
        filepath = db.fetchone(query, strict=True)
        return cls(filepath=filepath)

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> dict:
        filepath = safe_system_path(file_name=filepath)
        return pd.read_pickle(filepath)
