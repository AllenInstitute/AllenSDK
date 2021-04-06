import abc
from typing import Type

from pynwb import NWBFile


class DataObject(abc.ABC):
    def __init__(self, name: str, value: any):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @staticmethod
    @abc.abstractmethod
    def from_lims() -> Type["DataObject"]:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def from_json() -> Type["DataObject"]:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def from_nwb() -> Type["DataObject"]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_nwb(self, nwbfile: NWBFile):
        raise NotImplementedError()
