import pathlib


class CacheFileAttributes(object):
    """
    This class will contain the attributes of a remotely stored file
    so that they can easily and consistently be passed around between
    the methods making up the remote file cache and manifest classes

    Parameters
    ----------
    uri: str
        The full URI of the remote file
    version_id: str
        A string specifying the version of the file (probably calculated
        by S3)
    md5_checksum: str
        The (hexadecimal) md5 checksum of the file
    local_path: pathlib.Path
        The path to the location where the file's local copy should be stored
        (probably computed by the Manifest class)
    """

    def __init__(self,
                 uri: str,
                 version_id: str,
                 md5_checksum: str,
                 local_path: str):

        if not isinstance(uri, str):
            raise ValueError(f"uri must be str; got {type(uri)}")
        if not isinstance(version_id, str):
            raise ValueError(f"version_id must be str; got {type(version_id)}")
        if not isinstance(md5_checksum, str):
            raise ValueError(f"md5_checksum must be str; "
                             f"got {type(md5_checksum)}")
        if not isinstance(local_path, pathlib.Path):
            raise ValueError(f"local_path must be pathlib.Path; "
                             f"got {type(local_path)}")

        self._uri = uri
        self._version_id = version_id
        self._md5_checksum = md5_checksum
        self._local_path = local_path

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def version_id(self) -> str:
        return self._version_id

    @property
    def md5_checksum(self) -> str:
        return self._md5_checksum

    @property
    def local_path(self) -> pathlib.Path:
        return self._local_path

    def __str__(self):
        output = "CacheFileAttributes{\n"
        output += f"    uri: {self.uri}\n"
        output += f"    version_id: {self.version_id}\n"
        output += f"    md5_checkusm: {self.md5_checksum}\n"
        output += f"    local_path: {self.local_path}\n"
        output += "}\n"
        return output
