from functools import partial
from typing import Optional, List, Union
from pathlib import Path
import pandas as pd

from allensdk.api.warehouse_cache.cache import Cache
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .experiments_table import \
    ExperimentsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .sessions_table import \
    SessionsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import (  # noqa: E501
    BehaviorProjectLimsApi, BehaviorProjectCloudApi)
from allensdk.api.warehouse_cache.caching_utilities import \
    one_file_call_caching, call_caching
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .ophys_sessions_table import \
    BehaviorOphysSessionsTable
from allensdk.brain_observatory.behavior.behavior_project_cache \
    .project_cache_base import ProjectCacheBase


class VBOLimsCache(Cache):
    """
    A class that ineherits from the warehouse Cache and provides
    that functionality to VisualBehaviorOphysProjectCache
    """

    MANIFEST_VERSION = "0.0.1-alpha.3"
    OPHYS_SESSIONS_KEY = "ophys_sessions"
    BEHAVIOR_SESSIONS_KEY = "behavior_sessions"
    OPHYS_EXPERIMENTS_KEY = "ophys_experiments"
    OPHYS_CELLS_KEY = "ophys_cells"

    MANIFEST_CONFIG = {
        OPHYS_SESSIONS_KEY: {
            "spec": f"{OPHYS_SESSIONS_KEY}.csv",
            "parent_key": "BASEDIR",
            "typename": "file"
        },
        BEHAVIOR_SESSIONS_KEY: {
            "spec": f"{BEHAVIOR_SESSIONS_KEY}.csv",
            "parent_key": "BASEDIR",
            "typename": "file"
        },
        OPHYS_EXPERIMENTS_KEY: {
            "spec": f"{OPHYS_EXPERIMENTS_KEY}.csv",
            "parent_key": "BASEDIR",
            "typename": "file"
        },
        OPHYS_CELLS_KEY: {
            "spec": f"{OPHYS_CELLS_KEY}.csv",
            "parent_key": "BASEDIR",
            "typename": "file"
        }
    }


class VisualBehaviorOphysProjectCache(ProjectCacheBase):

    PROJECT_NAME = "visual-behavior-ophys"
    BUCKET_NAME = "visual-behavior-ophys-data"

    def __init__(
            self,
            fetch_api: Optional[Union[BehaviorProjectLimsApi,
                                      BehaviorProjectCloudApi]] = None,
            fetch_tries: int = 2,
            manifest: Optional[Union[str, Path]] = None,
            version: Optional[str] = None,
            cache: bool = True):
        """ Entrypoint for accessing visual behavior data. Supports
        access to summaries of session data and provides tools for
        downloading detailed session data (such as dff traces).

        Likely you will want to use a class constructor, such as `from_lims`,
        to initialize a VisualBehaviorOphysProjectCache, rather than calling
        this directly.


        Parameters
        ==========
        fetch_api :
            Used to pull data from remote sources, after which it is locally
            cached. Any object inheriting from BehaviorProjectBase is
            suitable. Current options are:
                BehaviorProjectLimsApi :: Fetches bleeding-edge data from the
                    Allen Institute"s internal database. Only works if you are
                    on our internal network.
        fetch_tries :
            Maximum number of times to attempt a download before giving up and
            raising an exception. Note that this is total tries, not retries.
            Default=2.
        manifest : str or Path
            full path at which manifest json will be stored. Defaults
            to "behavior_project_manifest.json" in the local directory.
        version : str
            version of manifest file. If this mismatches the version
            recorded in the file at manifest, an error will be raised.
            Defaults to the manifest version in the class.
        cache : bool
            Whether to write to the cache. Default=True.
        """

        super().__init__(fetch_api=fetch_api, fetch_tries=fetch_tries)

        if cache:
            manifest_ = manifest or "behavior_project_manifest.json"
        else:
            manifest_ = None

        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            if cache:
                self.cache = VBOLimsCache(manifest=manifest_,
                                          version=version,
                                          cache=cache)

    @classmethod
    def cloud_api_class(cls):
        return BehaviorProjectCloudApi

    @classmethod
    def lims_api_class(cls):
        return BehaviorProjectLimsApi

    def get_ophys_session_table(
            self,
            suppress: Optional[List[str]] = None,
            index_column: str = "ophys_session_id",
            as_df=True,
            include_behavior_data=True,
            passed_only=True) -> \
            Union[pd.DataFrame, BehaviorOphysSessionsTable]:
        """
        Return summary table of all ophys_session_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :param index_column: (default="ophys_session_id"). Column to index
        on, either
            "ophys_session_id" or "ophys_experiment_id".
            If index_column="ophys_experiment_id", then each row will only have
            one experiment id, of type int (vs. an array of 1>more).
        :type index_column: str
        :param as_df: whether to return as df or as BehaviorOphysSessionsTable
        :param include_behavior_data
            Whether to include behavior data
        :rtype: pd.DataFrame
        """
        if isinstance(self.fetch_api, BehaviorProjectCloudApi):
            return self.fetch_api.get_ophys_session_table()
        if self.cache is not None:
            path = self.cache.get_cache_path(None,
                                             self.cache.OPHYS_SESSIONS_KEY)
            ophys_sessions = one_file_call_caching(
                path,
                self.fetch_api.get_ophys_session_table,
                _write_json,
                lambda path: _read_json(path, index_name='ophys_session_id'))
        else:
            ophys_sessions = self.fetch_api.get_ophys_session_table()

        if include_behavior_data:
            # Merge behavior data in
            behavior_sessions_table = self.get_behavior_session_table(
                suppress=suppress, as_df=True, include_ophys_data=False)
            ophys_sessions = behavior_sessions_table.merge(
                ophys_sessions,
                left_index=True,
                right_on='behavior_session_id',
                suffixes=('_behavior', '_ophys'))

        sessions = BehaviorOphysSessionsTable(df=ophys_sessions,
                                              suppress=suppress,
                                              index_column=index_column)
        if passed_only:
            oet = self.get_ophys_experiment_table(passed_only=True)
            for i in sessions.table.index:
                sub_df = oet.query(f"ophys_session_id=={i}")
                values = list(set(sub_df["ophys_container_id"].values))
                values.sort()
                sessions.table.at[i, "ophys_container_id"] = values

        return sessions.table if as_df else sessions

    def get_ophys_experiment_table(
            self,
            suppress: Optional[List[str]] = None,
            as_df=True,
            passed_only=True) -> Union[pd.DataFrame, SessionsTable]:
        """
        Return summary table of all ophys_experiment_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :param as_df: whether to return as df or as SessionsTable
        :param passed_only: if True, return only experiments flagged as
                            'passed' and containers flagged as 'published'
                            (default=True)
        :rtype: pd.DataFrame
        """
        if isinstance(self.fetch_api, BehaviorProjectCloudApi):
            return self.fetch_api.get_ophys_experiment_table()
        if self.cache is not None:
            path = self.cache.get_cache_path(None,
                                             self.cache.OPHYS_EXPERIMENTS_KEY)
            experiments = one_file_call_caching(
                path,
                self.fetch_api.get_ophys_experiment_table,
                _write_json,
                lambda path: _read_json(path,
                                        index_name='ophys_experiment_id'))
        else:
            experiments = self.fetch_api.get_ophys_experiment_table()

        # Merge behavior data in
        behavior_sessions_table = self.get_behavior_session_table(
            suppress=suppress, as_df=True, include_ophys_data=False)
        experiments = behavior_sessions_table.merge(
            experiments, left_index=True, right_on='behavior_session_id',
            suffixes=('_behavior', '_ophys'))
        experiments = ExperimentsTable(df=experiments,
                                       suppress=suppress,
                                       passed_only=passed_only)
        return experiments.table if as_df else experiments

    def get_ophys_cells_table(self) -> pd.DataFrame:
        """
        Return summary table of all cells in this project cache
        :rtype: pd.DataFrame
        """
        if isinstance(self.fetch_api, BehaviorProjectCloudApi):
            return self.fetch_api.get_ophys_cells_table()
        if self.cache is not None:
            path = self.cache.get_cache_path(None,
                                             self.cache.OPHyS_CELLS_KEY)
            ophys_cells_table = one_file_call_caching(
                path,
                self.fetch_api.get_ophys_cells_table,
                _write_json,
                lambda path: _read_json(path,
                                        index_name='cell_roi_id'))
        else:
            ophys_cells_table = self.fetch_api.get_ophys_cells_table()

        return ophys_cells_table

    def get_behavior_session_table(
            self,
            suppress: Optional[List[str]] = None,
            as_df=True,
            include_ophys_data=True,
            passed_only=True) -> Union[pd.DataFrame, SessionsTable]:
        """
        Return summary table of all behavior_session_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :param as_df: whether to return as df or as SessionsTable
        :param include_ophys_data
            Whether to include ophys data
        :type suppress: list of str
        :rtype: pd.DataFrame
        """
        if isinstance(self.fetch_api, BehaviorProjectCloudApi):
            return self.fetch_api.get_behavior_session_table()
        if self.cache is not None:
            path = self.cache.get_cache_path(None,
                                             self.cache.BEHAVIOR_SESSIONS_KEY)
            sessions = one_file_call_caching(
                path,
                self.fetch_api.get_behavior_session_table,
                _write_json,
                lambda path: _read_json(path,
                                        index_name='behavior_session_id'))
        else:
            sessions = self.fetch_api.get_behavior_session_table()

        if include_ophys_data:
            ophys_session_table = self.get_ophys_session_table(
                suppress=suppress,
                as_df=False,
                include_behavior_data=False,
                passed_only=passed_only)
        else:
            ophys_session_table = None
        sessions = SessionsTable(df=sessions, suppress=suppress,
                                 fetch_api=self.fetch_api,
                                 ophys_session_table=ophys_session_table)

        return sessions.table if as_df else sessions

    def get_behavior_ophys_experiment(self, ophys_experiment_id: int,
                                      fixed: bool = False):
        """
        Note -- This method mocks the behavior of a cache. Future
        development will include an NWB reader to read from
        a true local cache (once nwb files are created).
        TODO: Using `fixed` will raise a NotImplementedError since there
        is no real cache.
        """
        if fixed:
            raise NotImplementedError
        fetch_session = partial(self.fetch_api.get_behavior_ophys_experiment,
                                ophys_experiment_id)
        return call_caching(
            fetch_session,
            lambda x: x,  # not writing anything
            lazy=False,  # can't actually read from file cache
            read=fetch_session
        )

    def get_behavior_session(self, behavior_session_id: int,
                             fixed: bool = False):
        """
        Note -- This method mocks the behavior of a cache. Future
        development will include an NWB reader to read from
        a true local cache (once nwb files are created).
        TODO: Using `fixed` will raise a NotImplementedError since there
        is no real cache.
        """
        if fixed:
            raise NotImplementedError

        fetch_session = partial(self.fetch_api.get_behavior_session,
                                behavior_session_id)
        return call_caching(
            fetch_session,
            lambda x: x,  # not writing anything
            lazy=False,  # can't actually read from file cache
            read=fetch_session
        )


def _write_json(path, df):
    """Wrapper to change the arguments for saving a pandas json
    dataframe so that it conforms to expectations of the internal
    cache methods. Can't use partial with the native `to_json` method
    because the dataframe is not yet created at the time we need to
    pass in the save method.
    Saves a dataframe in json format to `path`, in split orientation
    to save space on disk.
    Converts dates to seconds from epoch.
    NOTE: Date serialization is a big pain. Make sure if columns
    are being added, the _read_json is updated to properly deserialize
    them back to the expected format by adding them to `convert_dates`.
    In the future we could schematize this data using marshmallow
    or something similar."""
    df.to_json(path, orient="split", date_unit="s", date_format="epoch")


def _read_json(path, index_name: Optional[str] = None):
    """Reads a dataframe file written to the cache by _write_json."""
    df = pd.read_json(path, date_unit="s", orient="split",
                      convert_dates=["date_of_acquisition"])
    if index_name:
        df = df.rename_axis(index=index_name)
    return df
