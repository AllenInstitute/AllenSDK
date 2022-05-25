import abc
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import datetime
import copy

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.internal.core import DataFile
from allensdk.core import DataObject
from allensdk.core.pickle_utils import (
    load_and_sanitize_pickle,
    _sanitize_pickle_data)

# Query returns path to StimulusPickle file for given behavior session
BEHAVIOR_STIMULUS_FILE_QUERY_TEMPLATE = """
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


def from_json_cache_key(cls, stimulus_file_path: str):
    return hashkey(stimulus_file_path, cls.file_path_key())


def from_lims_cache_key(cls, db, behavior_session_id: int):
    return hashkey(behavior_session_id, cls.file_path_key())


class _StimulusFile(DataFile):
    """A DataFile which contains methods for accessing and loading visual
    behavior stimulus *.pkl files.

    This file type contains a number of parameters collected during a behavior
    session including information about stimulus presentations, rewards,
    trials, and timing for all of the above.
    """

    @classmethod
    def file_path_key(cls) -> str:
        """
        The key in the dict_repr that maps to the path
        to this StimulusFile's pickle file on disk.
        """
        raise NotImplementedError()

    def __init__(self, filepath: Union[str, Path]):
        self._sanitized_data = None
        super().__init__(filepath=filepath)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "_StimulusFile":
        filepath = dict_repr[cls.file_path_key()]
        return cls._from_json(stimulus_file_path=filepath)

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_json_cache_key)
    def _from_json(cls, stimulus_file_path: str) -> "_StimulusFile":
        return cls(filepath=stimulus_file_path)

    def to_json(self) -> Dict[str, str]:
        return {self.file_path_key(): str(self.filepath)}

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        behavior_session_id: Union[int, str]
    ) -> "_StimulusFile":
        raise NotImplementedError()

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> dict:
        filepath = safe_system_path(file_name=filepath)

        if pickle_path.name.endswith('gz'):
            open_method = gzip.open
        elif pickle_path.name.endswith('pkl'):
            open_method = open

        with open_method(pickle_path, 'rb') as in_file:
            return pickle.load(in_file, encoding='bytes')

    def _get_sanitized_data(self):
        """
        Save a copy of data with bytes converted to strings in the dict
        """
        self._sanitized_data = _sanitize_pickle(
                copy.deepcopy(self._data))

    @property
    def data(self) -> Any:  # pragma: no cover
        """
        To support backwards compatibility, this will return a copy
        of self._data that has been sanitized so that bytes are converted
        to string in the dict.
        """
        if self._sanitized_data is None:
            self._get_sanitized_data()
        return self._sanitized_data

    @property
    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        return len(self._data[b'intervalsms']) + 1


class BehaviorStimulusFile(_StimulusFile):

    @classmethod
    def file_path_key(cls) -> str:
        return "behavior_stimulus_file"

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        behavior_session_id: Union[int, str]
    ) -> "BehaviorStimulusFile":
        query = BEHAVIOR_STIMULUS_FILE_QUERY_TEMPLATE.format(
            behavior_session_id=behavior_session_id
        )
        filepath = db.fetchone(query, strict=True)
        return cls(filepath=filepath)

    @property
    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        return len(self._data[b'items'][b'behavior'][b'intervalsms']) + 1

    @property
    def date_of_acquisition(self) -> datetime.datetime:
        """
        Return the date_of_acquisition as a datetime.datetime.

        This will be read from self.data['start_time']
        """
        if b'start_time' not in self._data:
            raise KeyError(
                "No 'start_time' listed in pickle file "
                f"{self.filepath}")

        return copy.deepcopy(self._data[b'start_time'])

    @property
    def session_type(self) -> str:
        """
        Return the session type as read from the pickle file. This can
        be read either from

        data['items']['behavior']['params']['stage']
        or
        data['items']['behavior']['cl_params']['stage']

        if both are present and they disagree, raise an exception
        """
        param_value = None
        if b'params' in self._data[b'items'][b'behavior']:
            if b'stage' in self._data[b'items'][b'behavior'][b'params']:
                param_value = self._data[b'items'][b'behavior'][b'params'][b'stage']

        cl_value = None
        if b'cl_params' in self._data[b'items'][b'behavior']:
            if b'stage' in self._data[b'items'][b'behavior'][b'cl_params']:
                cl_value = self._data[b'items'][b'behavior'][b'cl_params'][b'stage']

        if cl_value is None and param_value is None:
            raise RuntimeError("Could not find stage in pickle file "
                               f"{self.filepath}")

        if param_value is None:
            desired_value = cl_value

        if cl_value is None:
            desired_value = param_value

        if cl_value != param_value:
            raise RuntimeError(
                "Conflicting session_types in pickle file "
                f"{self.filepath}\n"
                f"cl_params: {cl_value}\n"
                f"params: {param_value}\n")

        desired_value = param_value
        if isinstance(desired_value, bytes):
            desired_value = desired_value.decode('utf-8')
        return desired_value

    @property
    def session_duration(self) -> float:
        """
        Gets session duration in seconds

        Returns
        -------
        session duration in seconds
        """
        delta = self._data[b'stop_time'] - self._data[b'start_time']
        return delta.total_seconds()


class ReplayStimulusFile(_StimulusFile):

    @classmethod
    def file_path_key(cls) -> str:
        return "replay_stimulus_file"


class MappingStimulusFile(_StimulusFile):

    @classmethod
    def file_path_key(cls) -> str:
        return "mapping_stimulus_file"


class StimulusFileReadableInterface(abc.ABC):
    """Marks a data object as readable from stimulus file"""
    @classmethod
    @abc.abstractmethod
    def from_stimulus_file(
            cls,
            stimulus_file: BehaviorStimulusFile) -> "DataObject":
        """Populate a DataObject from the stimulus file

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        raise NotImplementedError()


class StimulusFileLookup(object):
    """
    A container class to carry around the StimulusFile(s) associated
    with a BehaviorSession
    """

    def __init__(self):
        self._values = dict()

    @property
    def behavior_stimulus_file(self) -> BehaviorStimulusFile:
        if 'behavior' not in self._values:
            raise ValueError("This StimulusFileLookup has no "
                             "BehaviorStimulusFile")
        return self._values['behavior']

    @behavior_stimulus_file.setter
    def behavior_stimulus_file(self, value: BehaviorStimulusFile):
        if not isinstance(value, BehaviorStimulusFile):
            raise ValueError("Trying to set behavior_stimulus_file to "
                             f"value of type {type(value)}; type should "
                             "be BehaviorStimulusFile")
        self._values['behavior'] = value

    @property
    def replay_stimulus_file(self) -> ReplayStimulusFile:
        if 'replay' not in self._values:
            raise ValueError("This StimulusFileLookup has no "
                             "ReplayStimulusFile")
        return self._values['replay']

    @replay_stimulus_file.setter
    def replay_stimulus_file(self, value: ReplayStimulusFile):
        if not isinstance(value, ReplayStimulusFile):
            raise ValueError("Trying to set replay_stimulus_file to "
                             f"value of type {type(value)}; type should "
                             "be ReplayStimulusFile")
        self._values['replay'] = value

    @property
    def mapping_stimulus_file(self) -> MappingStimulusFile:
        if 'mapping' not in self._values:
            raise ValueError("This StimulusFileLookup has no "
                             "MappingStimulusFile")
        return self._values['mapping']

    @mapping_stimulus_file.setter
    def mapping_stimulus_file(self, value: MappingStimulusFile):
        if not isinstance(value, MappingStimulusFile):
            raise ValueError("Trying to set mapping_stimulus_file to "
                             f"value of type {type(value)}; type should "
                             "be MappingStimulusFile")
        self._values['mapping'] = value


def stimulus_lookup_from_json(
        dict_repr: dict) -> StimulusFileLookup:
    """
    Load a lookup table of the stimulus files associated with a
    BehaviorSession from the dict representation of that session's
    session_data
    """
    lookup_table = StimulusFileLookup()
    if BehaviorStimulusFile.file_path_key() in dict_repr:
        stim = BehaviorStimulusFile.from_json(dict_repr=dict_repr)
        lookup_table.behavior_stimulus_file = stim
    if MappingStimulusFile.file_path_key() in dict_repr:
        stim = MappingStimulusFile.from_json(dict_repr=dict_repr)
        lookup_table.mapping_stimulus_file = stim
    if ReplayStimulusFile.file_path_key() in dict_repr:
        stim = ReplayStimulusFile.from_json(dict_repr=dict_repr)
        lookup_table.replay_stimulus_file = stim
    return lookup_table
