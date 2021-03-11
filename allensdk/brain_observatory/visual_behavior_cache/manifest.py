import json
import pathlib
import copy
from typing import Union
from allensdk.brain_observatory.visual_behavior_cache.utils import relative_path_from_uri  # noqa: E501
from allensdk.brain_observatory.visual_behavior_cache.file_attributes import CacheFileAttributes  # noqa: E501


class Manifest(object):
    """
    A class for loading and manipulating the on line manifest.json associated
    with a dataset release

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        The path to the directory where local copies of files will be stored
    """

    def __init__(self, cache_dir: Union[str, pathlib.Path]):
        if isinstance(cache_dir, str):
            self._cache_dir = pathlib.Path(cache_dir).resolve()
        elif isinstance(cache_dir, pathlib.Path):
            self._cache_dir = cache_dir.resolve()
        else:
            raise ValueError("cache_dir must be either a str "
                             "or a pathlib.Path; "
                             f"got {type(cache_dir)}")

        self._data = None
        self._version = None
        self._metadata_file_names = None

    @property
    def version(self):
        """
        The version of the dataset currently loaded
        """
        return self._version

    @property
    def metadata_file_names(self):
        """
        List of metadata file names associated with this dataset
        """
        return copy.deepcopy(self._metadata_file_names)

    def load(self, json_input):
        """
        Load a manifest.json

        Parameters
        ----------
        json_input:
            A ''.read()''-supporting file-like object containing
            a JSON document to be deserialized (i.e. same as the
            first argument to json.load)
        """
        self._data = json.load(json_input)
        if not isinstance(self._data, dict):
            raise ValueError("Expected to deserialize manifest into a dict; "
                             f"instead got {type(self._data)}")

        self._version = copy.deepcopy(self._data['dataset_version'])

        self._metadata_file_names = []
        for file_name in self._data['metadata_files'].keys():
            self._metadata_file_names.append(file_name)
        self._metadata_file_names.sort()

    def _create_file_attributes(self,
                                remote_path: str,
                                version_id: str,
                                file_hash: str) -> CacheFileAttributes:
        """
        Create the cache_file_attributes describing a file

        Parameters
        ----------
        remote_path: str
            The full URL to a file
        version_id: str
            The string specifying the version of the file
        file_hash: str
            The (hexadecimal) file hash of the file

        Returns
        -------
        CacheFileAttributes
        """

        local_dir = self._cache_dir / file_hash
        relative_path = relative_path_from_uri(remote_path)
        local_path = local_dir / relative_path

        obj = CacheFileAttributes(remote_path,
                                  version_id,
                                  file_hash,
                                  local_path)

        return obj

    def metadata_file_attributes(self,
                                 metadata_file_name: str) -> CacheFileAttributes:  # noqa: E501
        """
        Return the CacheFileAttributes associated with a metadata file

        Parameters
        ----------
        metadata_file_name: str
            Name of the metadata file. Must be in self.metadata_file_names

        Return
        ------
        CacheFileAttributes

        Raises
        ------
        RuntimeError
            If you try to run this method when self._data is None (meaning
            you haven't yet loaded a manifest.json)
        """
        if self._data is None:
            raise RuntimeError("You cannot retrieve "
                               "metadata_file_attributes;\n"
                               "you have not yet loaded a manifest.json file")

        if metadata_file_name not in self._metadata_file_names:
            raise ValueError(f"{metadata_file_name}\n"
                             "is not in self.metadata_file_names:\n"
                             f"{self._metadata_file_names}")

        file_data = self._data['metadata_files'][metadata_file_name]
        return self._create_file_attributes(file_data['uri'],
                                            file_data['s3_version'],
                                            file_data['file_hash'])

    def data_file_attributes(self, file_id) -> CacheFileAttributes:
        """
        Return the CacheFileAttributes associated with a data file

        Parameters
        ----------
        file_id:
            The identifier of the data file whose attributes are to be
            returned. Must be a key in self._data['data_files']

        Return
        ------
        CacheFileAttributes

        Raises
        ------
        RuntimeError
            If you try to run this method when self._data is None (meaning
            you haven't yet loaded a manifest.json file)
        """
        if self._data is None:
            raise RuntimeError("You cannot retrieve data_file_attributes;\n"
                               "you have not yet loaded a manifest.json file")

        if file_id not in self._data['data_files']:
            valid_keys = list(self._data['data_files'].keys())
            valid_keys.sort()
            raise ValueError(f"file_id: {file_id}\n"
                             "Is not a data file listed in manifest:\n"
                             f"{valid_keys}")

        file_data = self._data['data_files'][file_id]
        return self._create_file_attributes(file_data['uri'],
                                            file_data['s3_version'],
                                            file_data['file_hash'])
