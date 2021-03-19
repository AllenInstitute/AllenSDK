import pandas as pd
from typing import Iterable, Union
from pathlib import Path
import logging

from allensdk.brain_observatory.behavior.project_apis.abcs import (
    BehaviorProjectBase)
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_session import (
    BehaviorOphysExperiment)
from allensdk.api.cloud_cache.cloud_cache import S3CloudCache


class BehaviorProjectCloudApi(BehaviorProjectBase):
    """API for downloading data released on S3
     and returning tables.
    """
    def __init__(self, cache: S3CloudCache):
        """
        the passed cache should already have had `cache.load_manifest()`
        executed. With this satisfied, the attributes
          - metadata_file_names
          - file_id_column
        are populated
        """
        expected_metadata = set(["behavior_session_table.csv",
                                 "ophys_session_table.csv",
                                 "ophys_experiment_table.csv"])
        self.cache = cache
        if cache._manifest.metadata_file_names is None:
            raise RuntimeError("S3CloudCache object has no metadata "
                               "file names. BehaviorProjectCloudApi "
                               "expects a S3CloudCache passed which "
                               "has already run load_manifest()")
        cache_metadata = set(cache._manifest.metadata_file_names)
        if cache_metadata != expected_metadata:
            raise RuntimeError("expected S3CloudCache object to have "
                               f"metadata file names: {expected_metadata} "
                               f"but it has {cache_metadata}")
        self.logger = logging.getLogger("BehaviorProjectCloudApi")
        self._get_session_table()
        self._get_behavior_only_session_table()
        self._get_experiment_table()

    @staticmethod
    def from_s3_cache(cache_dir: Union[str, Path],
                      bucket_name: str) -> "BehaviorProjectCloudApi":
        cache = S3CloudCache(cache_dir, bucket_name)
        cache.load_latest_manifest()
        return BehaviorProjectCloudApi(cache)

    def get_behavior_only_session_data(
            self, behavior_session_id: int) -> BehaviorSession:
        """get a BehaviorSession by specifying behavior_session_id
        Parameters
        ----------
        behavior_session_id: int
            the id of the behavior_session
        Returns
        -------
        BehaviorSession
        """
        row = self._behavior_only_session_table.query(
                f"behavior_session_id=={behavior_session_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_only_session_table should have "
                               "1 and only 1 entry for a given "
                               "behavior_session_id. For "
                               f"{behavior_session_id} "
                               f" there are {row.shape[0]} entries.")
        data_path = self.cache.download_data(row.data_file_id.values[0])
        return BehaviorSession.from_nwb_path(str(data_path))

    def get_session_data(self, ophys_session_id: int
                         ) -> BehaviorOphysExperiment:
        """get a BehaviorOphysExperiment by specifying session_id
        Parameters
        ----------
        ophys_session_id: int
            the id of the ophys_session
        Returns
        -------
        BehaviorOphysExperiment
        """
        row = self._session_table.query(
                f"ophys_session_id=={ophys_session_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_session_table should have "
                               "1 and only 1 entry for a given "
                               f"ophys_session_id. For {ophys_session_id} "
                               f" there are {row.shape[0]} entries.")
        data_path = self.cache.download_data(row.data_file_id.values[0])
        return BehaviorOphysExperiment.from_nwb_path(str(data_path))

    def _get_session_table(self):
        session_table_path = self.cache.download_metadata(
                "ophys_session_table.csv")
        self._session_table = pd.read_csv(session_table_path)

    def get_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata."""
        return self._session_table

    def _get_behavior_only_session_table(self):
        session_table_path = self.cache.download_metadata(
                "behavior_session_table.csv")
        self._behavior_only_session_table = pd.read_csv(session_table_path)

    def get_behavior_only_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata."""
        return self._behavior_only_session_table

    def _get_experiment_table(self):
        experiment_table_path = self.cache.download_metadata(
                "ophys_experiment_table.csv")
        self._experiment_table = pd.read_csv(experiment_table_path)

    def get_experiment_table(self):
        return self._experiment_table

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        raise NotImplementedError()

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        raise NotImplementedError()
