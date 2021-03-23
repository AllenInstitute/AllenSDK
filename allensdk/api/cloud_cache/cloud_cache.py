from abc import ABC, abstractmethod
import os
import copy
import pathlib
import pandas as pd
import boto3
import semver
import tqdm
import re
from botocore import UNSIGNED
from botocore.client import Config
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.api.cloud_cache.manifest import Manifest
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501
from allensdk.api.cloud_cache.utils import file_hash_from_path  # noqa: E501
from allensdk.api.cloud_cache.utils import bucket_name_from_url  # noqa: E501
from allensdk.api.cloud_cache.utils import relative_path_from_url  # noqa: E501


class CloudCacheBase(ABC):
    """
    A class to handle the downloading and accessing of data served from a cloud
    storage system

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.
    """

    _bucket_name = None

    def __init__(self, cache_dir, project_name):
        os.makedirs(cache_dir, exist_ok=True)

        self._manifest = None
        self._cache_dir = cache_dir
        self._project_name = project_name
        self._manifest_file_names = self._list_all_manifests()

    @abstractmethod
    def _list_all_manifests(self) -> list:
        """
        Return a list of all of the file names of the manifests associated
        with this dataset
        """
        raise NotImplementedError()

    @property
    def latest_manifest_file(self) -> str:
        """parses available manifest files for semver string
        and returns the latest one
        self.manifest_file_names are assumed to be of the form
        '<anything>_v<semver_str>.json'

        Returns
        -------
        str
            the filename whose semver string is the latest one
        """
        vstrs = [s.split(".json")[0].split("_v")[-1]
                 for s in self.manifest_file_names]
        versions = [semver.VersionInfo.parse(v) for v in vstrs]
        imax = versions.index(max(versions))
        return self.manifest_file_names[imax]

    def load_latest_manifest(self):
        self.load_manifest(self.latest_manifest_file)

    @abstractmethod
    def _download_manifest(self,
                           manifest_name: str):
        """
        Download a manifest from the dataset

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        raise NotImplementedError()

    @abstractmethod
    def _download_file(self, file_attributes: CacheFileAttributes) -> bool:
        """
        Check if a file exists and is in the expected state.

        If it is, return True.

        If it is not, download the file, creating the directory
        where the file is to be stored if necessary.

        If the download is successful, return True.

        If the download fails (file hash does not match expectation),
        return False.

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Describes the file to download

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the path to the directory where the file is to be saved
            points to something that is not a directory.

        RuntimeError
            If it is not able to successfully download the file after
            10 iterations
        """
        raise NotImplementedError()

    @property
    def project_name(self) -> str:
        """
        The name of the project that this cache is accessing
        """
        return self._project_name

    @property
    def manifest_prefix(self) -> str:
        """
        On-line prefix for manifest files
        """
        return f'{self.project_name}/manifests/'

    @property
    def file_id_column(self) -> str:
        """
        The column in the metadata files used to uniquely
        identify data files
        """
        return self._manifest.file_id_column

    @property
    def version(self) -> str:
        """
        The version of the dataset currently loaded
        """
        return self._manifest.version

    @property
    def metadata_file_names(self) -> list:
        """
        List of metadata file names associated with this dataset
        """
        return self._manifest.metadata_file_names

    @property
    def manifest_file_names(self) -> list:
        """
        Sorted list of manifest file names associated with this
        dataset
        """
        return copy.deepcopy(self._manifest_file_names)

    def load_manifest(self, manifest_name: str):
        """
        Load a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        if manifest_name not in self.manifest_file_names:
            raise ValueError(f"manifest: {manifest_name}\n"
                             "is not one of the valid manifest names "
                             "for this dataset:\n"
                             f"{self.manifest_file_names}")

        filepath = os.path.join(self._cache_dir, manifest_name)
        if not os.path.exists(filepath):
            self._download_manifest(manifest_name)

        with open(filepath) as f:
            self._manifest = Manifest(
                cache_dir=self._cache_dir,
                json_input=f
            )

    def _file_exists(self, file_attributes: CacheFileAttributes) -> bool:
        """
        Given a CacheFileAttributes describing a file, assess whether or
        not that file exists locally and is valid (i.e. has the expected
        file hash)

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Description of the file to look for

        Returns
        -------
        bool
            True if the file exists and is valid; False otherwise

        Raises
        -----
        RuntimeError
            If file_attributes.local_path exists but is not a file.
            It would be unclear how the cache should proceed in this case.
        """

        if not file_attributes.local_path.exists():
            return False
        if not file_attributes.local_path.is_file():
            raise RuntimeError(f"{file_attributes.local_path}\n"
                               "exists, but is not a file;\n"
                               "unsure how to proceed")

        full_path = file_attributes.local_path.resolve()
        test_checksum = file_hash_from_path(full_path)
        if test_checksum != file_attributes.file_hash:
            return False

        return True

    def data_path(self, file_id) -> dict:
        """
        Return the local path to a data file, and test for the
        file's existence/validity

        Parameters
        ----------
        file_id:
            The unique identifier of the file to be accessed

        Returns
        -------
        dict

            'path' will be a pathlib.Path pointing to the file's location

            'exists' will be a boolean indicating if the file
            exists in a valid state

            'file_attributes' is a CacheFileAttributes describing the file
            in more detail

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        file_attributes = self._manifest.data_file_attributes(file_id)
        exists = self._file_exists(file_attributes)
        local_path = file_attributes.local_path
        output = {'local_path': local_path,
                  'exists': exists,
                  'file_attributes': file_attributes}

        return output

    def download_data(self, file_id) -> pathlib.Path:
        """
        Return the local path to a data file, downloading the file
        if necessary

        Parameters
        ----------
        file_id:
            The unique identifier of the file to be accessed

        Returns
        -------
        pathlib.Path
            The path indicating where the file is stored on the
            local system

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        super_attributes = self.data_path(file_id)
        file_attributes = super_attributes['file_attributes']
        self._download_file(file_attributes)
        return file_attributes.local_path

    def metadata_path(self, fname: str) -> dict:
        """
        Return the local path to a metadata file, and test for the
        file's existence/validity

        Parameters
        ----------
        fname: str
            The name of the metadata file to be accessed

        Returns
        -------
        dict

            'path' will be a pathlib.Path pointing to the file's location

            'exists' will be a boolean indicating if the file
            exists in a valid state

            'file_attributes' is a CacheFileAttributes describing the file
            in more detail

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        file_attributes = self._manifest.metadata_file_attributes(fname)
        exists = self._file_exists(file_attributes)
        local_path = file_attributes.local_path
        output = {'local_path': local_path,
                  'exists': exists,
                  'file_attributes': file_attributes}

        return output

    def download_metadata(self, fname: str) -> pathlib.Path:
        """
        Return the local path to a metadata file, downloading the
        file if necessary

        Parameters
        ----------
        fname: str
            The name of the metadata file to be accessed

        Returns
        -------
        pathlib.Path
            The path indicating where the file is stored on the
            local system

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        super_attributes = self.metadata_path(fname)
        file_attributes = super_attributes['file_attributes']
        self._download_file(file_attributes)
        return file_attributes.local_path

    def get_metadata(self, fname: str) -> pd.DataFrame:
        """
        Return a pandas DataFrame of metadata

        Parameters
        ----------
        fname: str
            The name of the metadata file to load

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        This method will check to see if the specified metadata file exists
        locally. If it does not, the method will download the file. Use
        self.metadata_path() to find where the file is stored
        """
        local_path = self.download_metadata(fname)
        return pd.read_csv(local_path)


class S3CloudCache(CloudCacheBase):
    """
    A class to handle the downloading and accessing of data served from
    an S3-based storage system

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system

    bucket_name: str
        for example, if bucket URI is 's3://mybucket' this value should be
        'mybucket'

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.
    """

    def __init__(self, cache_dir, bucket_name, project_name):
        self._manifest = None
        self._bucket_name = bucket_name

        super().__init__(cache_dir=cache_dir, project_name=project_name)

    _s3_client = None

    @property
    def s3_client(self):
        if self._s3_client is None:
            s3_config = Config(signature_version=UNSIGNED)
            self._s3_client = boto3.client('s3',
                                           config=s3_config)
        return self._s3_client

    def _list_all_manifests(self) -> list:
        """
        Return a list of all of the file names of the manifests associated
        with this dataset
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        subset_iterator = paginator.paginate(
            Bucket=self._bucket_name,
            Prefix=self.manifest_prefix
        )

        output = []
        for subset in subset_iterator:
            if 'Contents' in subset:
                for obj in subset['Contents']:
                    output.append(pathlib.Path(obj['Key']).name)

        output.sort()
        return output

    def _download_manifest(self,
                           manifest_name: str):
        """
        Download a manifest from the dataset

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """

        manifest_key = self.manifest_prefix + manifest_name
        response = self.s3_client.get_object(Bucket=self._bucket_name,
                                             Key=manifest_key)

        filepath = os.path.join(self._cache_dir, manifest_name)

        with open(filepath, 'wb') as f:
            for chunk in response['Body'].iter_chunks():
                f.write(chunk)

    def _download_file(self, file_attributes: CacheFileAttributes) -> bool:
        """
        Check if a file exists and is in the expected state.

        If it is, return True.

        If it is not, download the file, creating the directory
        where the file is to be stored if necessary.

        If the download is successful, return True.

        If the download fails (file hash does not match expectation),
        return False.

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Describes the file to download

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the path to the directory where the file is to be saved
            points to something that is not a directory.

        RuntimeError
            If it is not able to successfully download the file after
            10 iterations
        """

        local_path = file_attributes.local_path

        local_dir = pathlib.Path(safe_system_path(str(local_path.parents[0])))

        # make sure Windows references to Allen Institute
        # local networked file system get handled correctly
        local_path = pathlib.Path(safe_system_path(str(local_path)))

        # using os here rather than pathlib because safe_system_path
        # returns a str
        os.makedirs(local_dir, exist_ok=True)
        if not os.path.isdir(local_dir):
            raise RuntimeError(f"{local_dir}\n"
                               "is not a directory")

        bucket_name = bucket_name_from_url(file_attributes.url)
        obj_key = relative_path_from_url(file_attributes.url)

        n_iter = 0
        max_iter = 10  # maximum number of times to try download

        version_id = file_attributes.version_id

        pbar = None
        if not self._file_exists(file_attributes):
            response = self.s3_client.list_object_versions(Bucket=bucket_name,
                                                           Prefix=str(obj_key))
            object_info = [i for i in response["Versions"]
                           if i["VersionId"] == version_id][0]
            pbar = tqdm.tqdm(desc=object_info["Key"].split("/")[-1],
                             total=object_info["Size"],
                             unit_scale=True,
                             unit_divisor=1000.,
                             unit="MB")

        while not self._file_exists(file_attributes):
            response = self.s3_client.get_object(Bucket=bucket_name,
                                                 Key=str(obj_key),
                                                 VersionId=version_id)

            if 'Body' in response:
                with open(local_path, 'wb') as out_file:
                    for chunk in response['Body'].iter_chunks():
                        out_file.write(chunk)
                        pbar.update(len(chunk))

            n_iter += 1
            if n_iter > max_iter:
                pbar.close()
                raise RuntimeError("Could not download\n"
                                   f"{file_attributes}\n"
                                   "In {max_iter} iterations")
        if pbar is not None:
            pbar.close()
        return None


class LocalCache(CloudCacheBase):
    """A class to handle accessing of data that has already been downloaded
    locally

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.
    """
    def __init__(self, cache_dir, project_name):
        super().__init__(cache_dir=cache_dir, project_name=project_name)

    def _list_all_manifests(self) -> list:
        return [x for x in os.listdir(self._cache_dir)
                if re.fullmatch(".*_manifest_v.*.json", x)]

    def _download_manifest(self, manifest_name: str):
        raise NotImplementedError()

    def _download_file(self, file_attributes: CacheFileAttributes) -> bool:
        raise NotImplementedError()
