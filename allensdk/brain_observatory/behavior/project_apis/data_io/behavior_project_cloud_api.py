import pandas as pd
from typing import Iterable, Union
from pathlib import Path
import logging
import ast

from allensdk.brain_observatory.behavior.project_apis.abcs import (
    BehaviorProjectBase)
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
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
        expected_metadata = set(["behavior_session_table",
                                 "ophys_session_table",
                                 "ophys_experiment_table"])
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
                      bucket_name: str,
                      project_name: str) -> "BehaviorProjectCloudApi":
        """instantiates this object with a connection to an s3 bucket and/or
        a local cache related to that bucket.

        Parameters
        ----------
        cache_dir: str or pathlib.Path
            Path to the directory where data will be stored on the local system

        bucket_name: str
            for example, if bucket URI is 's3://mybucket' this value should be
            'mybucket'

        project_name: str
            the name of the project this cache is supposed to access. This
            project name is the first part of the prefix of the release data
            objects. I.e. s3://<bucket_name>/<project_name>/<object tree>

        Returns
        -------
        BehaviorProjectCloudApi instance

        """
        cache = S3CloudCache(cache_dir, bucket_name, project_name)
        cache.load_latest_manifest()
        return BehaviorProjectCloudApi(cache)

    def get_behavior_session(
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
        row = row.squeeze()
        has_file_id = not pd.isna(row[self.cache.file_id_column])
        if not has_file_id:
            # some entries in this table represent ophys sessions
            # which have a many-to-one mapping between nwb files
            # (1 per experiment) and behavior session.
            # in that case, the `file_id` column is nan.
            # this method returns an object which is just behavior data
            # which is shared by all experiments in 1 session
            # and so we just take the first ophys_experiment entry
            # to determine an appropriate nwb file to supply that information
            oeid = ast.literal_eval(row.ophys_experiment_id)[0]
            row = self._experiment_table.query(
                f"ophys_experiment_id=={oeid}").squeeze()
        data_path = self.cache.download_data(
                str(int(row[self.cache.file_id_column])))
        return BehaviorSession.from_nwb_path(str(data_path))

    def get_behavior_ophys_experiment(self, ophys_experiment_id: int
                                      ) -> BehaviorOphysExperiment:
        """get a BehaviorOphysExperiment by specifying ophys_experiment_id

        Parameters
        ----------
        ophys_experiment_id: int
            the id of the ophys_experiment

        Returns
        -------
        BehaviorOphysExperiment

        """
        row = self._experiment_table.query(
                f"ophys_experiment_id=={ophys_experiment_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_ophys_experiment_table should "
                               "have 1 and only 1 entry for a given "
                               f"ophys_experiment_id. For "
                               f"{ophys_experiment_id} "
                               f" there are {row.shape[0]} entries.")
        row = row.squeeze()
        data_path = self.cache.download_data(
                str(int(row[self.cache.file_id_column])))
        return BehaviorOphysExperiment.from_nwb_path(str(data_path))

    def _get_session_table(self) -> pd.DataFrame:
        session_table_path = self.cache.download_metadata(
                "ophys_session_table")
        self._session_table = pd.read_csv(session_table_path)

    def get_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table summarizing ophys_sessions
        and associated metadata.

        Notes
        -----
        - Each entry in this table represents the metadata of an ophys_session.
        Link to nwb-hosted files in the cache is had via the
        'ophys_experiment_id' column (can be a list)
        and experiment_table
        """
        return self._session_table

    def _get_behavior_only_session_table(self):
        session_table_path = self.cache.download_metadata(
                "behavior_session_table")
        self._behavior_only_session_table = pd.read_csv(session_table_path)

    def get_behavior_only_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with both behavior-only
        (BehaviorSession) and with-ophys (BehaviorOphysExperiment)
        sessions as entries.

        Notes
        -----
        - In the first case, provides a critical mapping of
        behavior_session_id to file_id, which the cache uses to find the
        nwb path in cache.
        - In the second case, provides a critical mapping of
        behavior_session_id to a list of ophys_experiment_id(s)
        which can be used to find file_id mappings in experiment_table
        see method get_behavior_session()
        - the BehaviorProjectCache calls this method through a method called
        get_behavior_session_table. The name of this method is a legacy shared
        with the behavior_project_lims_api and should be made consistent with
        the BehaviorProjectCache calling method.
        """
        return self._behavior_only_session_table

    def _get_experiment_table(self):
        experiment_table_path = self.cache.download_metadata(
                "ophys_experiment_table")
        self._experiment_table = pd.read_csv(experiment_table_path)

    def get_experiment_table(self):
        """returns a pd.DataFrame where each entry has a 1-to-1
        relation with an ophys experiment (i.e. imaging plane)

        Notes
        -----
        - the file_id column allows the underlying cache to link
        this table to a cache-hosted NWB file. There is a 1-to-1
        relation between nwb files and ophy experiments. See method
        get_behavior_ophys_experiment()
        """
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
