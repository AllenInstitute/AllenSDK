from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import os
import copy
import pathlib
import pandas as pd
import boto3
import semver
import tqdm
import re
import json
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

        # self._downloaded_data_path is where we will keep a JSONized
        # dict mapping paths to downloaded files to their file_hashes;
        # this will be used when determining if a downloaded file
        # can instead be a symlink
        c_path = pathlib.Path(self._cache_dir)
        self._downloaded_data_path = c_path / '_downloaded_data.json'

        self._project_name = project_name
        self._manifest_file_names = self._list_all_manifests()

    def _list_all_downloaded_manifests(self) -> list:
        """
        Return a list of all of the manifest files that have been
        downloaded for this dataset
        """
        return [x for x in os.listdir(self._cache_dir)
                if re.fullmatch(".*_manifest_v.*.json", x)]

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

    def _load_manifest(self, manifest_name: str) -> Manifest:
        """
        Load and return a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names

        Returns
        -------
        Manifest
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
            local_manifest = Manifest(
                cache_dir=self._cache_dir,
                json_input=f
            )
        return local_manifest

    def load_manifest(self, manifest_name: str):
        """
        Load a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        self._manifest = self._load_manifest(manifest_name)

    def _update_list_of_downloads(self,
                                  file_attributes: CacheFileAttributes
                                  ) -> None:
        """
        Update the local file that keeps track of files that have actually
        been downloaded to reflect a newly downloaded file.

        Parameters
        ----------
        file_attributes: CacheFileAttributes

        Returns
        -------
        None
        """
        if not file_attributes.local_path.exists():
            # This file does not exist; there is nothing to do
            return None

        if self._downloaded_data_path.exists():
            with open(self._downloaded_data_path, 'rb') as in_file:
                downloaded_data = json.load(in_file)
        else:
            downloaded_data = {}

        abs_path = str(file_attributes.local_path.resolve())
        downloaded_data[abs_path] = file_attributes.file_hash
        with open(self._downloaded_data_path, 'w') as out_file:
            out_file.write(json.dumps(downloaded_data,
                                      indent=2,
                                      sort_keys=True))
        return None

    def _check_for_identical_copy(self,
                                  file_attributes: CacheFileAttributes
                                  ) -> bool:
        """
        Check the manifest of files that have been locally downloaded to
        see if a file with an identical hash to the requested file has already
        been downloaded. If it has, create a symlink to the downloaded file
        at the requested file's localpath, update the manifest of downloaded
        files, and return True.

        Else return False

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            The file we are considering downloading

        Returns
        -------
        bool
        """
        if not self._downloaded_data_path.exists():
            return False

        with open(self._downloaded_data_path, 'rb') as in_file:
            available_files = json.load(in_file)

        matched_path = None
        for abs_path in available_files:
            if available_files[abs_path] == file_attributes.file_hash:
                matched_path = pathlib.Path(abs_path)
                break

        if matched_path is None:
            return False

        # double check that locally downloaded file still has
        # the expected hash
        candidate_hash = file_hash_from_path(matched_path)
        if candidate_hash != file_attributes.file_hash:
            return False

        local_parent = file_attributes.local_path.parent.resolve()
        if not local_parent.exists():
            os.makedirs(local_parent)

        file_attributes.local_path.symlink_to(matched_path.resolve())
        return True

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
        file_exists = False

        if file_attributes.local_path.exists():
            if not file_attributes.local_path.is_file():
                raise RuntimeError(f"{file_attributes.local_path}\n"
                                   "exists, but is not a file;\n"
                                   "unsure how to proceed")

            full_path = file_attributes.local_path.resolve()
            test_checksum = file_hash_from_path(full_path)
            if test_checksum == file_attributes.file_hash:
                file_exists = True

        if not file_exists:
            file_exists = self._check_for_identical_copy(file_attributes)

        return file_exists

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
        self._update_list_of_downloads(file_attributes)
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

    def _detect_changes(self,
                        filename_to_hash: dict) -> List[Tuple[str, str]]:
        """
        Assemble list of changes between two manifests

        Parameters
        ----------
        filename_to_hash: dict
            filename_to_hash[0] is a dict mapping file names to file hashes
            for manifest 0

            filename_to_hash[1] is a dict mapping file names to file hashes
            for manifest 1

        Returns
        -------
        List[Tuple[str, str]]
            List of changes between manifest 0 and manifest 1.

        Notes
        -----
        Changes are tuples of the form
        (fname, string describing how fname changed)

        e.g.

        ('data/f1.txt', 'data/f1.txt renamed data/f5.txt')
        ('data/f2.txt', 'data/f2.txt deleted')
        ('data/f3.txt', 'data/f3.txt created')
        ('data/f4.txt', 'data/f4.txt changed')
        """
        output = []
        n0 = set(filename_to_hash[0].keys())
        n1 = set(filename_to_hash[1].keys())
        all_file_names = n0.union(n1)

        hash_to_filename = {}
        for v in (0, 1):
            hash_to_filename[v] = {}
            for fname in filename_to_hash[v]:
                hash_to_filename[v][filename_to_hash[v][fname]] = fname

        for fname in all_file_names:
            delta = None
            if fname in filename_to_hash[0] and fname in filename_to_hash[1]:
                h0 = filename_to_hash[0][fname]
                h1 = filename_to_hash[1][fname]
                if h0 != h1:
                    delta = f'{fname} changed'
            elif fname in filename_to_hash[0]:
                h0 = filename_to_hash[0][fname]
                if h0 in hash_to_filename[1]:
                    f1 = hash_to_filename[1][h0]
                    delta = f'{fname} renamed {f1}'
                else:
                    delta = f'{fname} deleted'
            elif fname in filename_to_hash[1]:
                h1 = filename_to_hash[1][fname]
                if h1 not in hash_to_filename[0]:
                    delta = f'{fname} created'
            else:
                raise RuntimeError("should never reach this line")

            if delta is not None:
                output.append((fname, delta))

        return output

    def summarize_comparison(self,
                             manifest_0_name: str,
                             manifest_1_name: str
                             ) -> Dict[str, List[Tuple[str, str]]]:
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
        result: Dict[List[Tuple[str, str]]]
            result['data_changes'] lists changes to data files
            result['metadata_changes'] lists changes to metadata files

        Notes
        -----
        Changes are tuples of the form
        (fname, string describing how fname changed)

        e.g.

        ('data/f1.txt', 'data/f1.txt renamed data/f5.txt')
        ('data/f2.txt', 'data/f2.txt deleted')
        ('data/f3.txt', 'data/f3.txt created')
        ('data/f4.txt', 'data/f4.txt changed')
        """
        man0 = self._load_manifest(manifest_0_name)
        man1 = self._load_manifest(manifest_1_name)

        result = {}
        for (result_key,
             fname_list,
             fname_lookup) in zip(('metadata_changes', 'data_changes'),
                                  ((man0.metadata_file_names,
                                    man1.metadata_file_names),
                                   (man0.file_id_values,
                                    man1.file_id_values)),
                                  ((man0.metadata_file_attributes,
                                    man1.metadata_file_attributes),
                                   (man0.data_file_attributes,
                                    man1.data_file_attributes))):

            filename_to_hash = {}
            for version in (0, 1):
                filename_to_hash[version] = {}
                for file_id in fname_list[version]:
                    obj = fname_lookup[version](file_id)
                    file_name = relative_path_from_url(obj.url)
                    file_name = '/'.join(file_name.split('/')[1:])
                    filename_to_hash[version][file_name] = obj.file_hash
            changes = self._detect_changes(filename_to_hash)
            result[result_key] = changes
        return result


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
        return self._list_all_downloaded_manifests()

    def _download_manifest(self, manifest_name: str):
        raise NotImplementedError()

    def _download_file(self, file_attributes: CacheFileAttributes) -> bool:
        raise NotImplementedError()
