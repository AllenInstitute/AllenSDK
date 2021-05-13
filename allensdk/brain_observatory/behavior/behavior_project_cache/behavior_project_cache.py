from functools import partial
from typing import Optional, List, Union
from pathlib import Path
import pandas as pd
import logging

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
from allensdk.core.authentication import DbCredentials


class VBOLimsCache(Cache):
    """
    A class that ineherits from the warehouse Cache and provides
    that functionality to VisualBehaviorOphysProjectCache
    """

    MANIFEST_VERSION = "0.0.1-alpha.3"
    OPHYS_SESSIONS_KEY = "ophys_sessions"
    BEHAVIOR_SESSIONS_KEY = "behavior_sessions"
    OPHYS_EXPERIMENTS_KEY = "ophys_experiments"

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
        }
    }


class VisualBehaviorOphysProjectCache(object):

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

        --- NOTE ---
        Because NWB files are not currently supported for this project (as of
        11/2019), this cache will not actually save any files of session data
        to the local machine. Only summary tables will be saved to the local
        cache. File retrievals for specific sessions will be handled by
        the fetch api used for the Session object, and cached in-memory
        only to enable fast retrieval for subsequent calls.

        If you are looping over session objects, be sure to clean up
        your memory when it is not needed by calling `cache_clear` from
        your session object.

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
        if cache:
            manifest_ = manifest or "behavior_project_manifest.json"
        else:
            manifest_ = None

        self.fetch_api = fetch_api
        self.cache = None

        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            if cache:
                self.cache = VBOLimsCache(manifest=manifest_,
                                          version=version,
                                          cache=cache)
        self.fetch_tries = fetch_tries
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def manifest(self):
        if self.cache is None:
            api_name = type(self.fetch_api).__name__
            raise NotImplementedError(f"A {type(self).__name__} "
                                      f"based on {api_name} "
                                      "does not have an accessible manifest "
                                      "property")
        return self.cache.manifest

    @classmethod
    def from_s3_cache(cls, cache_dir: Union[str, Path],
                      bucket_name: str = "visual-behavior-ophys-data",
                      project_name: str = "visual-behavior-ophys"
                      ) -> "VisualBehaviorOphysProjectCache":
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
        VisualBehaviorOphysProjectCache instance

        """
        fetch_api = BehaviorProjectCloudApi.from_s3_cache(
                cache_dir, bucket_name, project_name,
                ui_class_name=cls.__name__)
        return cls(fetch_api=fetch_api)

    @classmethod
    def from_local_cache(cls, cache_dir: Union[str, Path],
                         project_name: str = "visual-behavior-ophys"
                         ) -> "VisualBehaviorOphysProjectCache":
        """instantiates this object with a local cache.

        Parameters
        ----------
        cache_dir: str or pathlib.Path
            Path to the directory where data will be stored on the local system

        project_name: str
            the name of the project this cache is supposed to access. This
            project name is the first part of the prefix of the release data
            objects. I.e. s3://<bucket_name>/<project_name>/<object tree>

        Returns
        -------
        VisualBehaviorOphysProjectCache instance

        """
        fetch_api = BehaviorProjectCloudApi.from_local_cache(
                cache_dir, project_name,
                ui_class_name=cls.__name__)
        return cls(fetch_api=fetch_api)

    @classmethod
    def from_lims(cls, manifest: Optional[Union[str, Path]] = None,
                  version: Optional[str] = None,
                  cache: bool = False,
                  fetch_tries: int = 2,
                  lims_credentials: Optional[DbCredentials] = None,
                  mtrain_credentials: Optional[DbCredentials] = None,
                  host: Optional[str] = None,
                  scheme: Optional[str] = None,
                  asynchronous: bool = True,
                  data_release_date: Optional[str] = None
                  ) -> "VisualBehaviorOphysProjectCache":
        """
        Construct a VisualBehaviorOphysProjectCache with a lims api. Use this
        method to create a  VisualBehaviorOphysProjectCache instance rather
        than calling VisualBehaviorOphysProjectCache directly.

        Parameters
        ==========
        manifest : str or Path
            full path at which manifest json will be stored
        version : str
            version of manifest file. If this mismatches the version
            recorded in the file at manifest, an error will be raised.
        cache : bool
            Whether to write to the cache
        fetch_tries : int
            Maximum number of times to attempt a download before giving up and
            raising an exception. Note that this is total tries, not retries
        lims_credentials : DbCredentials
            Optional credentials to access LIMS database.
            If not set, will look for credentials in environment variables.
        mtrain_credentials: DbCredentials
            Optional credentials to access mtrain database.
            If not set, will look for credentials in environment variables.
        host : str
            Web host for the app_engine. Currently unused. This argument is
            included for consistency with EcephysProjectCache.from_lims.
        scheme : str
            URI scheme, such as "http". Currently unused. This argument is
            included for consistency with EcephysProjectCache.from_lims.
        asynchronous : bool
            Whether to fetch from web asynchronously. Currently unused.
        data_release_date: str
            Use to filter tables to only include data released on date
            ie 2021-03-25
        Returns
        =======
        VisualBehaviorOphysProjectCache
            VisualBehaviorOphysProjectCache instance with a LIMS fetch API
        """
        if host and scheme:
            app_kwargs = {"host": host, "scheme": scheme,
                          "asynchronous": asynchronous}
        else:
            app_kwargs = None
        fetch_api = BehaviorProjectLimsApi.default(
            lims_credentials=lims_credentials,
            mtrain_credentials=mtrain_credentials,
            data_release_date=data_release_date,
            app_kwargs=app_kwargs)
        return cls(fetch_api=fetch_api, manifest=manifest, version=version,
                   cache=cache, fetch_tries=fetch_tries)

    def _cache_not_implemented(self, method_name: str) -> None:
        """
        Raise a NotImplementedError explaining that method_name
        does not exist for VisualBehaviorOphysProjectCache
        that does not have a fetch_api based on LIMS
        """
        msg = f"Method {method_name} does not exist for this "
        msg += f"{type(self).__name__}, which is based on "
        msg += f"{type(self.fetch_api).__name__}"
        raise NotImplementedError(msg)

    def construct_local_manifest(self) -> None:
        """
        Construct the local file used to determine if two files are
        duplicates of each other or not. Save it into the expected
        place in the cache. (You will see a warning if the cache
        thinks that you need to run this method).
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('construct_local_manifest')
        self.fetch_api.cache.construct_local_manifest()

    def compare_manifests(self,
                          manifest_0_name: str,
                          manifest_1_name: str
                          ) -> str:
        """
        Compare two manifests from this dataset. Return a dict
        containing the list of metadata and data files that changed
        between them

        Note: this assumes that manifest_0 predates manifest_1

        Parameters
        ----------
        manifest_0_name: str

        manifest_1_name: str

        Returns
        -------
        str
            A string summarizing all of the changes going from
            manifest_0 to manifest_1
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('compare_manifests')
        return self.fetch_api.cache.compare_manifests(manifest_0_name,
                                                      manifest_1_name)

    def load_latest_manifest(self) -> None:
        """
        Load the manifest corresponding to the most up to date
        version of the dataset.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('load_latest_manifest')
        self.fetch_api.cache.load_latest_manifest()

    def latest_downloaded_manifest_file(self) -> str:
        """
        Return the name of the most up to date data manifest
        available on your local system.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('latest_downloaded_manifest_file')
        return self.fetch_api.cache.latest_downloaded_manifest_file

    def latest_manifest_file(self) -> str:
        """
        Return the name of the most up to date data manifest
        corresponding to this dataset, checking in the cloud
        if this is a cloud-backed cache.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('latest_manifest_file')
        return self.fetch_api.cache.latest_manifest_file

    def load_manifest(self, manifest_name: str):
        """
        Load a specific versioned manifest for this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('load_manifest')
        self.fetch_api.load_manifest(manifest_name)

    def list_all_downloaded_manifests(self) -> list:
        """
        Return a sorted list of the names of the manifest files
        that have been downloaded to this cache.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('list_all_downloaded_manifests')
        return self.fetch_api.cache.list_all_downloaded_manifests()

    def list_manifest_file_names(self) -> list:
        """
        Return a sorted list of the names of the manifest files
        associated with this dataset.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('list_manifest_file_names')
        return self.fetch_api.cache.manifest_file_names

    def current_manifest(self) -> Union[None, str]:
        """
        Return the name of the dataset manifest currently being
        used by this cache.
        """
        if not isinstance(self.fetch_api, BehaviorProjectCloudApi):
            self._cache_not_implemented('current_manifest')
        return self.fetch_api.cache.current_manifest

    def get_ophys_session_table(
            self,
            suppress: Optional[List[str]] = None,
            index_column: str = "ophys_session_id",
            as_df=True,
            include_behavior_data=True) -> \
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

        return sessions.table if as_df else sessions

    def get_ophys_experiment_table(
            self,
            suppress: Optional[List[str]] = None,
            as_df=True) -> Union[pd.DataFrame, SessionsTable]:
        """
        Return summary table of all ophys_experiment_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :param as_df: whether to return as df or as SessionsTable
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
                                       suppress=suppress)
        return experiments.table if as_df else experiments

    def get_behavior_session_table(
            self,
            suppress: Optional[List[str]] = None,
            as_df=True,
            include_ophys_data=True) -> Union[pd.DataFrame, SessionsTable]:
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
                include_behavior_data=False)
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
