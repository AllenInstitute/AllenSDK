import pytest
import pathlib
import hashlib
import numpy as np
import allensdk.brain_observatory.visual_behavior_cache.utils as utils


def test_bucket_name_from_uri():

    uri = 'https://dummy_bucket.s3.amazonaws.com/txt_file.txt?versionId="jklaafdaerew"'  # noqa: E501
    bucket_name = utils.bucket_name_from_uri(uri)
    assert bucket_name == "dummy_bucket"

    uri = 'https://dummy_bucket/txt_file.txt?versionId="jklaafdaerew"'
    with pytest.warns(UserWarning):
        bucket_name = utils.bucket_name_from_uri(uri)
    assert bucket_name is None


def test_relative_path_from_uri():
    uri = 'https://dummy_bucket.s3.amazonaws.com/my/dir/txt_file.txt?versionId="jklaafdaerew"'  # noqa: E501
    relative_path = utils.relative_path_from_uri(uri)
    assert relative_path == pathlib.Path('my/dir/txt_file.txt')


def test_file_hash_from_path(tmpdir):

    rng = np.random.RandomState(881)
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    fname = tmpdir / 'hash_dummy.txt'
    with open(fname, 'w') as out_file:
        for ii in range(10):
            out_file.write(''.join(rng.choice(alphabet, size=10)))
            out_file.write('\n')

    hasher = hashlib.blake2b()
    with open(fname, 'rb') as in_file:
        chunk = in_file.read(7)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(7)

    ans = utils.file_hash_from_path(fname)
    assert ans == hasher.hexdigest()
