from functools import partial
from typing import Type, Optional, List, Union
from pathlib import Path
import pandas as pd
import logging

from allensdk.api.cache import Cache

from allensdk.brain_observatory.behavior.project_apis.data_io import (
    BehaviorProjectLimsApi)
from allensdk.brain_observatory.behavior.project_apis.abcs import (
    BehaviorProjectBase)
from allensdk.api.caching_utilities import one_file_call_caching, call_caching
from allensdk.core.authentication import DbCredentials

BehaviorProjectApi = Type[BehaviorProjectBase]


class BehaviorProjectCache(Cache):

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

    def __init__(
            self,
            fetch_api: Optional[BehaviorProjectApi] = None,
            fetch_tries: int = 2,
            manifest: Optional[Union[str, Path]] = None,
            version: Optional[str] = None,
            cache: bool = True):
        """ Entrypoint for accessing visual behavior data. Supports
        access to summaries of session data and provides tools for
        downloading detailed session data (such as dff traces).

        Likely you will want to use a class constructor, such as `from_lims`,
        to initialize a BehaviorProjectCache, rather than calling this
        directly.

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
        version_ = version or self.MANIFEST_VERSION

        super().__init__(manifest=manifest_, version=version_, cache=cache)
        self.fetch_api = fetch_api
        self.fetch_tries = fetch_tries
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_lims(cls, manifest: Optional[Union[str, Path]] = None,
                  version: Optional[str] = None,
                  cache: bool = False,
                  fetch_tries: int = 2,
                  lims_credentials: Optional[DbCredentials] = None,
                  mtrain_credentials: Optional[DbCredentials] = None,
                  host: Optional[str] = None,
                  scheme: Optional[str] = None,
                  asynchronous: bool = True) -> "BehaviorProjectCache":
        """
        Construct a BehaviorProjectCache with a lims api. Use this method
        to create a  BehaviorProjectCache instance rather than calling
        BehaviorProjectCache directly.

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
        Returns
        =======
        BehaviorProjectCache
            BehaviorProjectCache instance with a LIMS fetch API
        """
        if host and scheme:
            app_kwargs = {"host": host, "scheme": scheme,
                          "asynchronous": asynchronous}
        else:
            app_kwargs = None
        fetch_api = BehaviorProjectLimsApi.default(
                        lims_credentials=lims_credentials,
                        mtrain_credentials=mtrain_credentials,
                        app_kwargs=app_kwargs)
        return cls(fetch_api=fetch_api, manifest=manifest, version=version,
                   cache=cache, fetch_tries=fetch_tries)

    def get_session_table(
            self,
            suppress: Optional[List[str]] = None,
            by: str = "ophys_session_id") -> pd.DataFrame:
        """
        Return summary table of all ophys_session_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :param by: (default="ophys_session_id"). Column to index on, either
            "ophys_session_id" or "ophys_experiment_id".
            If by="ophys_experiment_id", then each row will only have one
            experiment id, of type int (vs. an array of 1>more).
        :type by: str
        :rtype: pd.DataFrame
        """
        if self.cache:
            path = self.get_cache_path(None, self.OPHYS_SESSIONS_KEY)
            sessions = one_file_call_caching(
                path,
                self.fetch_api.get_session_table,
                _write_json, _read_json)
            sessions.set_index("ophys_session_id")
        else:
            sessions = self.fetch_api.get_session_table()
        if suppress:
            sessions.drop(columns=suppress, inplace=True, errors="ignore")

        # Possibly explode and reindex
        if by == "ophys_session_id":
            pass
        elif by == "ophys_experiment_id":
            sessions = (sessions.reset_index()
                        .explode("ophys_experiment_id")
                        .set_index("ophys_experiment_id"))
        else:
            self.logger.warning(
                f"Invalid value for `by`, '{by}', passed to get_session_table."
                " Valid choices for `by` are 'ophys_experiment_id' and "
                "'ophys_session_id'.")
        return sessions

    def add_manifest_paths(self, manifest_builder):
        manifest_builder = super().add_manifest_paths(manifest_builder)
        for key, config in self.MANIFEST_CONFIG.items():
            manifest_builder.add_path(key, **config)
        return manifest_builder

    def get_experiment_table(
            self,
            suppress: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return summary table of all ophys_experiment_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :rtype: pd.DataFrame
        """
        if self.cache:
            path = self.get_cache_path(None, self.OPHYS_EXPERIMENTS_KEY)
            experiments = one_file_call_caching(
                path,
                self.fetch_api.get_experiment_table,
                _write_json, _read_json)
            experiments.set_index("ophys_experiment_id")
        else:
            experiments = self.fetch_api.get_experiment_table()
        if suppress:
            experiments.drop(columns=suppress, inplace=True, errors="ignore")
        return experiments

    def get_behavior_session_table(
            self,
            suppress: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return summary table of all behavior_session_ids in the database.
        :param suppress: optional list of columns to drop from the resulting
            dataframe.
        :type suppress: list of str
        :rtype: pd.DataFrame
        """

        if self.cache:
            path = self.get_cache_path(None, self.BEHAVIOR_SESSIONS_KEY)
            sessions = one_file_call_caching(
                path,
                self.fetch_api.get_behavior_only_session_table,
                _write_json, _read_json)
            sessions.set_index("behavior_session_id")
        else:
            sessions = self.fetch_api.get_behavior_only_session_table()
        sessions = sessions.rename(columns={"genotype": "full_genotype"})
        if suppress:
            sessions.drop(columns=suppress, inplace=True, errors="ignore")
        return sessions

    def get_session_data(self, ophys_experiment_id: int, fixed: bool = False):
        """
        Note -- This method mocks the behavior of a cache. Future
        development will include an NWB reader to read from
        a true local cache (once nwb files are created).
        TODO: Using `fixed` will raise a NotImplementedError since there
        is no real cache.
        """
        if fixed:
            raise NotImplementedError
        fetch_session = partial(self.fetch_api.get_session_data,
                                ophys_experiment_id)
        return call_caching(
            fetch_session,
            lambda x: x,        # not writing anything
            lazy=False,         # can't actually read from file cache
            read=fetch_session
        )

    def get_behavior_session_data(self, behavior_session_id: int,
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

        fetch_session = partial(self.fetch_api.get_behavior_only_session_data,
                                behavior_session_id)
        return call_caching(
            fetch_session,
            lambda x: x,       # not writing anything
            lazy=False,        # can't actually read from file cache
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
    df.reset_index(inplace=True)
    df.to_json(path, orient="split", date_unit="s", date_format="epoch")


def _read_json(path):
    """Reads a dataframe file written to the cache by _write_json."""
    df = pd.read_json(path, date_unit="s", orient="split",
                      convert_dates=["date_of_acquisition"])
    return df
