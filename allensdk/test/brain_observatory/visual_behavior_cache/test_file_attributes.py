import pytest
import pathlib
from allensdk.brain_observatory.visual_behavior_cache.file_attributes import CacheFileAttributes  # noqa: E501


def test_cache_file_attributes():
    attr = CacheFileAttributes(uri='http://my/uri',
                               version_id='aaabbb',
                               file_hash='12345',
                               local_path=pathlib.Path('/my/local/path'))

    assert attr.uri == 'http://my/uri'
    assert attr.version_id == 'aaabbb'
    assert attr.file_hash == '12345'
    assert attr.local_path == pathlib.Path('/my/local/path')

    # test that the correct ValueErrors are raised
    # when you pass invalid arguments

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(uri=5.0,
                                   version_id='aaabbb',
                                   file_hash='12345',
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "uri must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(uri='http://my/uri/',
                                   version_id=5.0,
                                   file_hash='12345',
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "version_id must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(uri='http://my/uri/',
                                   version_id='aaabbb',
                                   file_hash=5.0,
                                   local_path=pathlib.Path('/my/local/path'))

    msg = "file_hash must be str; got <class 'float'>"
    assert context.value.args[0] == msg

    with pytest.raises(ValueError) as context:
        attr = CacheFileAttributes(uri='http://my/uri/',
                                   version_id='aaabbb',
                                   file_hash='12345',
                                   local_path='/my/local/path')

    msg = "local_path must be pathlib.Path; got <class 'str'>"
    assert context.value.args[0] == msg
