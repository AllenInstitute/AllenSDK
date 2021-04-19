import abc
from typing import Any, Union
from pathlib import Path


class DataFile(abc.ABC):
    """An abstract class that prototypes methods for accessing internal
    data files.

    These data files contain information necessary to sucessfully instantiate
    one or many `DataObject`(s).

    External users should ignore this class (and subclasses) as as they
    will only ever be using `from_nwb()` and `to_nwb()` `DataObject` methods.
    """

    def __init__(self, filepath: Union[str, Path]):  # pragma: no cover
        self._filepath: str = str(filepath)
        self._data = self.load_data(filepath=filepath)

    @property
    def data(self) -> Any:  # pragma: no cover
        return self._data

    @property
    def filepath(self) -> str:  # pragma: no cover
        return self._filepath

    @classmethod
    @abc.abstractmethod
    def from_json(cls) -> "DataFile":  # pragma: no cover
        # Example:
        # filepath = dict_repr["my_data_file_path"]
        # return cls.instantiate(filepath=filepath)
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self) -> dict:  # pragma: no cover
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_lims(cls) -> "DataFile":  # pragma: no cover
        # Example:
        # query = """SELECT my_file FROM some_lims_table"""
        # filepath = dbconn.fetchone(query, strict=True)
        # return cls.instantiate(filepath=filepath)
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def load_data(filepath: Union[str, Path]) -> Any:  # pragma: no cover
        raise NotImplementedError()
