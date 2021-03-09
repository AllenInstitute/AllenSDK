from typing import Optional
import warnings
import urllib.parse as url_parse
import pathlib
import hashlib


def bucket_name_from_uri(uri: str) -> Optional[str]:
    """
    Read in a URI and return the name of the AWS S3 bucket it points towards

    Parameters
    ----------
    uri: str
        A generic URI

    Returns
    -------
    str
        An AWS S3 bucket name. Note: if 's3.amazonaws.com' does not occur in
        the URI, this method will return None and emit a warning.
    """
    s3_pattern = '.s3.amazonaws.com'
    url_params = url_parse.urlparse(uri)
    if s3_pattern not in url_params.netloc:
        warnings.warn(f"{s3_pattern} does not occur in URI {uri}")
        return None
    return url_params.netloc.replace(s3_pattern, '')


def relative_path_from_uri(uri: str) -> pathlib.Path:
    """
    Read in a URI and return the relative path of the object

    Parameters
    ----------
    uri: str
        The URI of the object whose path you want

    Returns
    -------
    pathlib.Path:
        Relative path of the object
    """
    url_params = url_parse.urlparse(uri)
    return pathlib.Path(url_params.path[1:])


def md5_hash_from_path(file_path: str) -> str:
    """
    Return the hexadecimal md5 checksum for a file

    Parameters
    ----------
    file_path: str
        path to a file

    Returns
    -------
    str:
        The md5 checksum (hexadecimal) of the file
    """
    md5sum = hashlib.md5()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(1000000)
        while len(chunk) > 0:
            md5sum.update(chunk)
            chunk = in_file.read(1000000)
    return md5sum.hexdigest()
