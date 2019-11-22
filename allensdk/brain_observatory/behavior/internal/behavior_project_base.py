from abc import ABC, abstractmethod
from typing import Iterable

from allensdk.brain_observatory.behavior.behavior_ophys_session import (
    BehaviorOphysSession)
from allensdk.brain_observatory.behavior.behavior_data_session import (
    BehaviorDataSession)
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
            self, behavior_session_id: int) -> BehaviorDataSession:
        """Returns a BehaviorDataSession object that contains methods to
        analyze a single behavior session.
        :param behavior_session_id: id that corresponds to a behavior session
        :type behavior_session_id: int
        :rtype: BehaviorDataSession
        """
        pass

    @abstractmethod
    def get_behavior_only_session_table(self) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        pass

    @abstractmethod
    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        pass
