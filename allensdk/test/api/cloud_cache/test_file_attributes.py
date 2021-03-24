import platform
import pytest
import pathlib
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501


def test_cache_file_attributes():
    attr = CacheFileAttributes(url='http://my/url',
                               version_id='aaabbb',
                               file_hash='12345',
                               local_path=pathlib.Path('/my/local/path'))

    assert attr.url == 'http://my/url'
    assert attr.version_id == 'aaabbb'
    assert attr.file_hash == '12345'
    assert attr.local_path == pathlib.Path('/my/local/path')

    # test that the correct ValueErrors are raised
    # when you pass invalid arguments

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(url=5.0,
                                   version_id='aaabbb',
                                   file_hash='12345',
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "url must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(url='http://my/url/',
                                   version_id=5.0,
                                   file_hash='12345',
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "version_id must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(url='http://my/url/',
                                   version_id='aaabbb',
                                   file_hash=5.0,
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "file_hash must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(url='http://my/url/',
                                   version_id='aaabbb',
                                   file_hash='12345',
                                   local_path='/my/local/path')

    msg = "local_path must be pathlib.Path; got <class 'str'>"
    assert context.value.args[0] == msg


def test_str():
    """
    Test the string representation of CacheFileParameters
    """
    attr = CacheFileAttributes(url='http://my/url',
                               version_id='aaabbb',
                               file_hash='12345',
                               local_path=pathlib.Path('/my/local/path'))

    s = f'{attr}'
    assert "CacheFileParameters{" in s
    assert '"file_hash": "12345"' in s
    assert '"url": "http://my/url"' in s
    assert '"version_id": "aaabbb"' in s
    if platform.system().lower() != 'windows':
        assert '"local_path": "/my/local/path"' in s
