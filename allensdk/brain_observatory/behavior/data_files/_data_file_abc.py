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
        self._data = self.load_data(filepath=self._filepath)

    @property
    def data(self) -> Any:  # pragma: no cover
        return self._data

    @property
    def filepath(self) -> str:  # pragma: no cover
        return self._filepath

    @classmethod
    @abc.abstractmethod
    def from_json(cls, dict_repr: dict) -> "DataFile":  # pragma: no cover
        """Populates a DataFile from a JSON compatible dict (likely parsed by
        argschema)

        Returns
        -------
        DataFile:
            An instantiated DataFile which has `data` and `filepath` properties
        """
        # Example:
        # filepath = dict_repr["my_data_file_path"]
        # return cls.instantiate(filepath=filepath)
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self) -> dict:  # pragma: no cover
        """Given an already populated DataFile, return the dict that
        when used with the `from_json()` classmethod would produce the same
        DataFile

        Returns
        -------
        dict:
            The JSON (in dict form) that would produce the DataFile.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_lims(cls) -> "DataFile":  # pragma: no cover
        """Populate a DataFile from an internal database (likely LIMS)

        Returns
        -------
        DataFile:
            An instantiated DataFile which has `data` and `filepath` properties
        """
        # Example:
        # query = """SELECT my_file FROM some_lims_table"""
        # filepath = dbconn.fetchone(query, strict=True)
        # return cls.instantiate(filepath=filepath)
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def load_data(filepath: Union[str, Path]) -> Any:  # pragma: no cover
        """Given a filepath (that is meant to by read by the DataFile type),
        load the contents of the file into a Python type.
        (dict, DataFrame, list, etc...)

        Parameters
        ----------
        filepath : Union[str, Path]
            The filepath that the DataFile class should load.

        Returns
        -------
        Any
            A Python data type that has been parsed/loaded from the provided
            filepath.
        """
        raise NotImplementedError()
