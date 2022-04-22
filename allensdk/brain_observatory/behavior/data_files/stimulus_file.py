import abc
from typing import Dict, Union
from pathlib import Path

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.internal.core import DataFile
from allensdk.core import DataObject

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


class _NumFramesMixin(object):
    """
    Mixin to implement num_frames for a generic (i.e. non-behavior)
    StimulusFile
    """

    def _validate_frame_data(self) -> None:
        """
        Check that self.data['intervalsms'] is present and
        that self.data['items']['behavior']['intervalsms'] is empty
        """
        msg = ""
        if 'intervalsms' not in self.data:
            msg += "self.data['intervalsms'] not present\n"
        if "items" in self.data:
            if "behavior" in self.data["items"]:
                if "intervalsms" in self.data["items"]["behavior"]:
                    val = self.data["items"]["behavior"]["intervalsms"]
                    if len(val) > 0:
                        msg += ("len(self.data['items']['behavior']"
                                f"['intervalsms'] == {len(val)}; "
                                "expected zero\n")
        if len(msg) > 0:
            full_msg = f"When getting num_frames from {type(self)}\n"
            full_msg += msg
            full_msg += f"\nfilepath: {self.filepath}"
            raise RuntimeError(full_msg)

        return None

    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        self._validate_frame_data()
        return len(self.data['intervalsms']) + 1


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
        return pd.read_pickle(filepath)

    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        raise NotImplementedError()


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

    def _validate_frame_data(self):
        """
        Make sure that self.data['intervalsms'] does not exist and that
        self.data['items']['behavior']['intervalsms'] does exist.
        """
        msg = ""
        if "intervalsms" in self.data:
            msg += "self.data['intervalsms'] present; did not expect that\n"
        if "items" not in self.data:
            msg += "self.data['items'] not present\n"
        else:
            if "behavior" not in self.data["items"]:
                msg += "self.data['items']['behavior'] not present\n"
            else:
                if "intervalsms" not in self.data["items"]["behavior"]:
                    msg += ("self.data['items']['behavior']['intervalsms'] "
                            "not present\n")

        if len(msg) > 0:
            full_msg = f"When getting num_frames from {type(self)}\n"
            full_msg += msg
            full_msg += f"\nfilepath: {self.filepath}"
            raise RuntimeError(full_msg)

        return None

    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        self._validate_frame_data()
        return len(self.data['items']['behavior']['intervalsms']) + 1


class ReplayStimulusFile(_NumFramesMixin, _StimulusFile):

    @classmethod
    def file_path_key(cls) -> str:
        return "replay_stimulus_file"


class MappingStimulusFile(_NumFramesMixin, _StimulusFile):

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
