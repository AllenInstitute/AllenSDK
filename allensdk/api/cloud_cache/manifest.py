from typing import Dict, List, Any
import json
import pathlib
from typing import Union
from allensdk.api.cloud_cache.utils import relative_path_from_url  # noqa: E501
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501


class Manifest(object):
    """
    A class for loading and manipulating the online manifest.json associated
    with a dataset release

    Each Manifest instance should represent the data for 1 and only 1
    manifest.json file.

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        The path to the directory where local copies of files will be stored
    json_input:
        A ''.read()''-supporting file-like object containing
        a JSON document to be deserialized (i.e. same as the
        first argument to json.load)
    """

    def __init__(self,
                 cache_dir: Union[str, pathlib.Path],
                 json_input):
        if isinstance(cache_dir, str):
            self._cache_dir = pathlib.Path(cache_dir).resolve()
        elif isinstance(cache_dir, pathlib.Path):
            self._cache_dir = cache_dir.resolve()
        else:
            raise ValueError("cache_dir must be either a str "
                             "or a pathlib.Path; "
                             f"got {type(cache_dir)}")

        self._data: Dict[str, Any] = json.load(json_input)
        if not isinstance(self._data, dict):
            raise ValueError("Expected to deserialize manifest into a dict; "
                             f"instead got {type(self._data)}")
        self._project_name: str = self._data["project_name"]
        self._version: str = self._data['manifest_version']
        self._file_id_column: str = self._data['metadata_file_id_column_name']
        self._data_pipeline: str = self._data["data_pipeline"]

        self._metadata_file_names: List[str] = [
            file_name for file_name in self._data['metadata_files']
        ]
        self._metadata_file_names.sort()

        self._file_id_values: List[Any] = [ii for ii in
                                           self._data['data_files'].keys()]
        self._file_id_values.sort()

    @property
    def project_name(self):
        """
        The name of the project whose data and metadata files this
        manifest tracks.
        """
        return self._project_name

    @property
    def version(self):
        """
        The version of the dataset currently loaded
        """
        return self._version

    @property
    def file_id_column(self):
        """
        The column in the metadata files used to uniquely
        identify data files
        """
        return self._file_id_column

    @property
    def metadata_file_names(self):
        """
        List of metadata file names associated with this dataset
        """
        return self._metadata_file_names

    @property
    def file_id_values(self):
        """
        List of valid file_id values
        """
        return self._file_id_values

    def _create_file_attributes(self,
                                remote_path: str,
                                version_id: str,
                                file_hash: str) -> CacheFileAttributes:
        """
        Create the cache_file_attributes describing a file.
        This method does the work of assigning a local_path to a remote file.

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

        # Paths should be built like:
        # {cache_dir} / {project_name}-{manifest_version} / relative_path
        # Ex: my_cache_dir/visual-behavior-ophys-1.0.0/behavior_sessions/etc...

        project_dir_name = f"{self._project_name}-{self._version}"
        project_dir = self._cache_dir / project_dir_name

        # The convention of the data release tool is to have all
        # relative_paths from remote start with the project name which
        # we want to remove since we already specified a project directory
        relative_path = relative_path_from_url(remote_path)
        shaved_rel_path = "/".join(relative_path.split("/")[1:])

        local_path = project_dir / shaved_rel_path

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

        ValueError
            If the metadata_file_name is not a valid option
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
        return self._create_file_attributes(file_data['url'],
                                            file_data['version_id'],
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

        ValueError
            If the file_id is not a valid option
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
        return self._create_file_attributes(file_data['url'],
                                            file_data['version_id'],
                                            file_data['file_hash'])
