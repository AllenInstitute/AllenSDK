from typing import Protocol
from abc import abstractmethod


class SupportsStr(Protocol):
    """Classes that support the __str__ method"""
    @abstractmethod
    def __str__(self) -> str:
        pass
