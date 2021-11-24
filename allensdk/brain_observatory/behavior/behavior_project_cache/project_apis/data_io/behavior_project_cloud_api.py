import pandas as pd
from typing import Iterable, Union, List, Optional
from pathlib import Path
import logging
import ast
import semver

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.abcs import BehaviorProjectBase  # noqa: E501
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)
from allensdk.api.cloud_cache.cloud_cache import (
    S3CloudCache, LocalCache, StaticLocalCache)


# [min inclusive, max exclusive)
MANIFEST_COMPATIBILITY = ["1.0.0", "2.0.0"]


class BehaviorCloudCacheVersionException(Exception):
    pass


def version_check(manifest_version: str,
                  data_pipeline_version: str,
                  cmin: str = MANIFEST_COMPATIBILITY[0],
                  cmax: str = MANIFEST_COMPATIBILITY[1]):
    mver_parsed = semver.VersionInfo.parse(manifest_version)
    cmin_parsed = semver.VersionInfo.parse(cmin)
    cmax_parsed = semver.VersionInfo.parse(cmax)
    if (mver_parsed < cmin_parsed) | (mver_parsed >= cmax_parsed):
        estr = (f"the manifest has manifest_version {manifest_version} but "
                "this version of AllenSDK is compatible only with manifest "
                f"versions {cmin} <= X < {cmax}. \n"
                "Consider using a version of AllenSDK closer to the version "
                f"used to release the data: {data_pipeline_version}")
        raise BehaviorCloudCacheVersionException(estr)


def literal_col_eval(df: pd.DataFrame,
                     columns: List[str] = ["ophys_experiment_id",
                                           "ophys_container_id",
                                           "driver_line"]) -> pd.DataFrame:
    def converter(x):
        if isinstance(x, str):
            x = ast.literal_eval(x)
        return x

    for column in columns:
        if column in df.columns:
            df.loc[df[column].notnull(), column] = \
                df[column][df[column].notnull()].apply(converter)
    return df


class BehaviorProjectCloudApi(BehaviorProjectBase):
    """API for downloading data released on S3 and returning tables.

    Parameters
    ----------
    cache: S3CloudCache
        an instantiated S3CloudCache object, which has already run
        `self.load_manifest()` which populates the columns:
          - metadata_file_names
          - file_id_column
    skip_version_check: bool
        whether to skip the version checking of pipeline SDK version
        vs. running SDK version, which may raise Exceptions. (default=False)
    local: bool
        Whether to operate in local mode, where no data will be downloaded
        and instead will be loaded from local
    """
    def __init__(
        self,
        cache: Union[S3CloudCache, LocalCache, StaticLocalCache],
        skip_version_check: bool = False,
        local: bool = False
    ):

        self.cache = cache
        self.skip_version_check = skip_version_check
        self._local = local
        self.load_manifest()

    def load_manifest(self, manifest_name: Optional[str] = None):
        """
        Load the specified manifest file into the CloudCache

        Parameters
        ----------
        manifest_name: Optional[str]
            Name of manifest file to load. If None, load latest
            (default: None)
        """
        if manifest_name is None:
            self.cache.load_last_manifest()
        else:
            self.cache.load_manifest(manifest_name)

        expected_metadata = set(["behavior_session_table",
                                 "ophys_session_table",
                                 "ophys_experiment_table",
                                 "ophys_cells_table"])

        if self.cache._manifest.metadata_file_names is None:
            raise RuntimeError("S3CloudCache object has no metadata "
                               "file names. BehaviorProjectCloudApi "
                               "expects a S3CloudCache passed which "
                               "has already run load_manifest()")
        cache_metadata = set(self.cache._manifest.metadata_file_names)

        if cache_metadata != expected_metadata:
            raise RuntimeError("expected S3CloudCache object to have "
                               f"metadata file names: {expected_metadata} "
                               f"but it has {cache_metadata}")

        if not self.skip_version_check:
            data_sdk_version = [i for i in self.cache._manifest._data_pipeline
                                if i['name'] == "AllenSDK"][0]["version"]
            version_check(self.cache._manifest.version, data_sdk_version)

        #    version_check(self.cache._manifest._data_pipeline)
        self.logger = logging.getLogger("BehaviorProjectCloudApi")
        self._get_ophys_session_table()
        self._get_behavior_session_table()
        self._get_ophys_experiment_table()
        self._get_ophys_cells_table()

    @staticmethod
    def from_s3_cache(cache_dir: Union[str, Path],
                      bucket_name: str,
                      project_name: str,
                      ui_class_name: str) -> "BehaviorProjectCloudApi":
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

        ui_class_name: str
            Name of user interface class (used to populate error messages)

        Returns
        -------
        BehaviorProjectCloudApi instance

        """
        cache = S3CloudCache(cache_dir,
                             bucket_name,
                             project_name,
                             ui_class_name=ui_class_name)
        return BehaviorProjectCloudApi(cache)

    @staticmethod
    def from_local_cache(
        cache_dir: Union[str, Path],
        project_name: str,
        ui_class_name: str,
        use_static_cache: bool = False
    ) -> "BehaviorProjectCloudApi":
        """instantiates this object with a local cache.

        Parameters
        ----------
        cache_dir: str or pathlib.Path
            Path to the directory where data will be stored on the local system

        project_name: str
            the name of the project this cache is supposed to access. This
            project name is the first part of the prefix of the release data
            objects. I.e. s3://<bucket_name>/<project_name>/<object tree>

        ui_class_name: str
            Name of user interface class (used to populate error messages)

        Returns
        -------
        BehaviorProjectCloudApi instance

        """
        if use_static_cache:
            cache = StaticLocalCache(
                cache_dir,
                project_name,
                ui_class_name=ui_class_name
            )
        else:
            cache = LocalCache(
                cache_dir,
                project_name,
                ui_class_name=ui_class_name
            )
        return BehaviorProjectCloudApi(cache, local=True)

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

        Notes
        -----
        entries in the _behavior_session_table represent
        (1) ophys_sessions which have a many-to-one mapping between nwb files
        and behavior sessions. (file_id is NaN)
        AND
        (2) behavior only sessions, which have a one-to-one mapping with
        nwb files. (file_id is not Nan)
        In the case of (1) this method returns an object which is just behavior
        data which is shared by all experiments in 1 session. This is extracted
        from the nwb file for the first-listed ophys_experiment.

        """
        row = self._behavior_session_table.query(
                f"behavior_session_id=={behavior_session_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_session_table should have "
                               "1 and only 1 entry for a given "
                               "behavior_session_id. For "
                               f"{behavior_session_id} "
                               f" there are {row.shape[0]} entries.")
        row = row.squeeze()
        has_file_id = not pd.isna(row[self.cache.file_id_column])
        if not has_file_id:
            oeid = row.ophys_experiment_id[0]
            row = self._ophys_experiment_table.query(f"index=={oeid}")
        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)
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
        row = self._ophys_experiment_table.query(
                f"index=={ophys_experiment_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_ophys_experiment_table should "
                               "have 1 and only 1 entry for a given "
                               f"ophys_experiment_id. For "
                               f"{ophys_experiment_id} "
                               f" there are {row.shape[0]} entries.")
        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)
        return BehaviorOphysExperiment.from_nwb_path(
            str(data_path))

    def _get_ophys_session_table(self):
        session_table_path = self._get_metadata_path(
            fname="ophys_session_table")
        df = literal_col_eval(pd.read_csv(session_table_path))
        self._ophys_session_table = df.set_index("ophys_session_id")

    def get_ophys_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table summarizing ophys_sessions
        and associated metadata.

        Notes
        -----
        - Each entry in this table represents the metadata of an ophys_session.
        Link to nwb-hosted files in the cache is had via the
        'ophys_experiment_id' column (can be a list)
        and experiment_table
        """
        return self._ophys_session_table

    def _get_behavior_session_table(self):
        session_table_path = self._get_metadata_path(
            fname='behavior_session_table')
        df = literal_col_eval(pd.read_csv(session_table_path))
        self._behavior_session_table = df.set_index("behavior_session_id")

    def get_behavior_session_table(self) -> pd.DataFrame:
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
        which can be used to find file_id mappings in ophys_experiment_table
        see method get_behavior_session()
        """
        return self._behavior_session_table

    def _get_ophys_experiment_table(self):
        experiment_table_path = self._get_metadata_path(
            fname="ophys_experiment_table")
        df = literal_col_eval(pd.read_csv(experiment_table_path))
        self._ophys_experiment_table = df.set_index("ophys_experiment_id")

    def _get_ophys_cells_table(self):
        ophys_cells_table_path = self._get_metadata_path(
            fname="ophys_cells_table")
        df = literal_col_eval(pd.read_csv(ophys_cells_table_path))
        # NaN's for invalid cells force this to float, push to int
        df['cell_specimen_id'] = pd.array(df['cell_specimen_id'],
                                          dtype="Int64")
        self._ophys_cells_table = df.set_index("cell_roi_id")

    def get_ophys_cells_table(self):
        return self._ophys_cells_table

    def get_ophys_experiment_table(self):
        """returns a pd.DataFrame where each entry has a 1-to-1
        relation with an ophys experiment (i.e. imaging plane)

        Notes
        -----
        - the file_id column allows the underlying cache to link
        this table to a cache-hosted NWB file. There is a 1-to-1
        relation between nwb files and ophy experiments. See method
        get_behavior_ophys_experiment()
        """
        return self._ophys_experiment_table

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        raise NotImplementedError()

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        raise NotImplementedError()

    def _get_metadata_path(self, fname: str):
        if self._local:
            path = self._get_local_path(fname=fname)
        else:
            path = self.cache.download_metadata(fname=fname)
        return path

    def _get_data_path(self, file_id: str):
        if self._local:
            data_path = self._get_local_path(file_id=file_id)
        else:
            data_path = self.cache.download_data(file_id=file_id)
        return data_path

    def _get_local_path(self, fname: Optional[str] = None, file_id:
                        Optional[str] = None):
        if fname is None and file_id is None:
            raise ValueError('Must pass either fname or file_id')

        if fname is not None and file_id is not None:
            raise ValueError('Must pass only one of fname or file_id')

        if fname is not None:
            path = self.cache.metadata_path(fname=fname)
        else:
            path = self.cache.data_path(file_id=file_id)

        exists = path['exists']
        local_path = path['local_path']
        if not exists:
            raise FileNotFoundError(f'You started a cache without a '
                                    f'connection to s3 and {local_path} is '
                                    'not already on your system')
        return local_path
