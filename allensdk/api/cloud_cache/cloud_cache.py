import os
import copy
import io
import pathlib
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.api.cloud_cache.manifest import Manifest
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501
from allensdk.api.cloud_cache.utils import file_hash_from_path  # noqa: E501
from allensdk.api.cloud_cache.utils import bucket_name_from_uri  # noqa: E501
from allensdk.api.cloud_cache.utils import relative_path_from_uri  # noqa: E501


class CloudCache(object):
    """
    A class to handle the downloading and accessing of data served from a cloud
    storage system

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system
    """

    _bucket_name = None

    def __init__(self, cache_dir):
        self._manifest = Manifest(cache_dir)
        self._s3_client = None
        self._manifest_file_names = self._list_all_manifests()

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
    def metadata_flie_names(self) -> list:
        """
        List of metadata file names associated with this dataset
        """
        return self._manifest.metadata_file_names

    @property
    def s3_client(self):
        if self._s3_client is None:
            s3_config = Config(signature_version=UNSIGNED)
            self._s3_client = boto3.client('s3',
                                           config=s3_config)
        return self._s3_client

    @property
    def manifest_file_names(self) -> list:
        """
        Sorted list of manifest file names associated with this
        dataset
        """
        return copy.deepcopy(self._manifest_file_names)

    def _list_all_manifests(self) -> list:
        """
        Return a list of all of the file names of the manifests associated
        with this dataset
        """
        output = []
        continuation_token = None
        keep_going = True
        while keep_going:
            if continuation_token is not None:
                subset = self.s3_client.list_objects_v2(Bucket=self._bucket_name,  # noqa: E501
                                                        Prefix='manifests/',
                                                        ContinuationToken=continuation_token)  # noqa: E501
            else:
                subset = self.s3_client.list_objects_v2(Bucket=self._bucket_name,  # noqa: E501
                                                        Prefix='manifests/')

            if 'Contents' in subset:
                for obj in subset['Contents']:
                    output.append(pathlib.Path(obj['Key']).name)

            if 'NextContinuationToken' in subset:
                continuation_token = subset['NextContinuationToken']
            else:
                keep_going = False

        output.sort()
        return output

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

        manifest_key = 'manifests/' + manifest_name
        response = self.s3_client.get_object(Bucket=self._bucket_name,
                                             Key=manifest_key)
        with io.BytesIO() as stream:
            for chunk in response['Body'].iter_chunks():
                stream.write(chunk)
            stream.seek(0)
            self._manifest.load(stream)

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
        boolean

        Raises
        ------
        RuntimeError
            If the path to the directory where the file is to be saved
            points to something that is not a directory.
        """

        local_path = file_attributes.local_path
        local_dir = safe_system_path(local_path.parents[0])

        # using os here rather than pathlib because safe_system_path
        # returns a str
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        if not os.path.isdir(local_dir):
            raise RuntimeError(f"{local_dir}\n"
                               "is not a directory")

        bucket_name = bucket_name_from_uri(file_attributes.uri)
        obj_key = relative_path_from_uri(file_attributes.uri)

        n_iter = 0
        max_iter = 10  # maximum number of times to try download

        version_id = file_attributes.version_id

        while not self._file_exists(file_attributes):
            response = self.s3_client.get_object(Bucket=bucket_name,
                                                 Key=str(obj_key),
                                                 VersionId=version_id)

            if 'Body' in response:
                with open(local_path, 'wb') as out_file:
                    for chunk in response['Body'].iter_chunks():
                        out_file.write(chunk)

            n_iter += 1
            if n_iter > max_iter:
                return False
        return True

    def data_path(self, file_id) -> pathlib.Path:
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
        file_attributes = self._manifest.data_file_attributes(file_id)
        is_valid = self._download_file(file_attributes)
        if not is_valid:
            raise RuntimeError("Unable to download file\n"
                               f"file_id: {file_id}\n"
                               f"{file_attributes}")

        return file_attributes.local_path

    def metadata_path(self, fname: str) -> pathlib.Path:
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
        file_attributes = self._manifest.metadata_file_attributes(fname)
        is_valid = self._download_file(file_attributes)
        if not is_valid:
            raise RuntimeError("Unable to download file\n"
                               f"file_id: {fname}\n"
                               f"{file_attributes}")

        return file_attributes.local_path

    def metadata(self, fname: str) -> pd.DataFrame:
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
        local_path = self.metadata_path(fname)
        return pd.read_csv(local_path)
