import json
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.brain_observatory.behavior.sync import get_sync_data
from allensdk.brain_observatory.behavior.data_files import DataFile


# Query returns path to sync timing file associated with ophys experiment
SYNC_FILE_QUERY_TEMPLATE = """
    SELECT wkf.storage_directory || wkf.filename AS sync_file
    FROM ophys_experiments oe
    JOIN ophys_sessions os ON oe.ophys_session_id = os.id
    JOIN well_known_files wkf ON wkf.attachable_id = os.id
    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id
    WHERE wkf.attachable_type = 'OphysSession'
    AND wkft.name = 'OphysRigSync'
    AND oe.id = {ophys_experiment_id};
"""


def from_json_cache_key(cls, dict_repr: dict):
    return hashkey(json.dumps(dict_repr))


def from_lims_cache_key(cls, db, ophys_experiment_id: int):
    return hashkey(ophys_experiment_id)


class SyncFile(DataFile):
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains global timing information for different data
    streams collected during a behavior + ophys session.
    """

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def from_json(cls, dict_repr: dict) -> "SyncFile":
        filepath = dict_repr["sync_file"]
        return cls(filepath=filepath)

    def to_json(self) -> Dict[str, str]:
        return {"sync_file": str(self.filepath)}

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        ophys_experiment_id: Union[int, str]
    ) -> "SyncFile":
        query = SYNC_FILE_QUERY_TEMPLATE.format(
            ophys_experiment_id=ophys_experiment_id
        )
        filepath = db.fetchone(query, strict=True)
        return cls(filepath=filepath)

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> dict:
        filepath = safe_system_path(file_name=filepath)
        return get_sync_data(sync_path=filepath)
