from abc import ABC, abstractmethod
from typing import Iterable

from allensdk.brain_observatory.behavior.behavior_ophys_session import (
    BehaviorOphysSession)
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
import pandas as pd


class BehaviorProjectBase(ABC):
    @abstractmethod
    def get_session_data(self, ophys_session_id: int) -> BehaviorOphysSession:
        """Returns a BehaviorOphysSession object that contains methods
        to analyze a single behavior+ophys session.
        :param ophys_session_id: id that corresponds to a behavior session
        :type ophys_session_id: int
        :rtype: BehaviorOphysSession
        """
        pass

    @abstractmethod
    def get_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata."""
        pass

    @abstractmethod
    def get_behavior_only_session_data(
            self, behavior_session_id: int) -> BehaviorSession:
        """Returns a BehaviorSession object that contains methods to
        analyze a single behavior session.
        :param behavior_session_id: id that corresponds to a behavior session
        :type behavior_session_id: int
        :rtype: BehaviorSession
        """
        pass

    @abstractmethod
    def get_behavior_only_session_table(self) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.
        :rtype: pd.DataFrame
        """
        pass
