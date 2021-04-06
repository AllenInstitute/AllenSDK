import abc


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

    @abc.abstractmethod
    def from_lims(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def from_json(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def from_nwb(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_nwb(self):
        raise NotImplementedError()
