import os
import hashlib
import io
import sys

import pytest

import allensdk.brain_observatory.ecephys.copy_utility.__main__ as cu


def test_hash_file(tmpdir_factory):
    tempdir = str(tmpdir_factory.mktemp('ecephys_copy_utility_test_hash_file'))
    path = os.path.join(tempdir, 'afile.txt')

    st = 'hello world'
    with open(path, 'wb') as f:
        f.write(st.encode())

    hasher_cls = hashlib.sha256
    obtained = cu.hash_file(path, hasher_cls)

    h = hasher_cls()
    h.update(st.encode())
    expected = h.digest()
    assert expected == obtained


@pytest.mark.parametrize('use_rsync', [True, False])
@pytest.mark.parametrize('make_parent_dirs', [True, False])
@pytest.mark.parametrize("chmod", [777, 775, 755, None])
def test_copy_file_entry(tmpdir_factory, use_rsync, make_parent_dirs, chmod):

    mac_or_linux = sys.platform.startswith('darwin') or sys.platform.startswith('linux')
    if use_rsync and not mac_or_linux:
        pytest.skip()

    tempdir = str(tmpdir_factory.mktemp('ecephys_copy_utility_test_copy_file_entry'))
    spath = os.path.join(tempdir, 'afile.txt')
    dpath = os.path.join(tempdir, 'bfile.txt')

    with open(spath, 'w') as sf:
        sf.write('foo')

    cu.copy_file_entry(spath, dpath, use_rsync, make_parent_dirs, chmod)
    
    with open(dpath, 'r') as df:
        assert df.read() == 'foo'

    get_human_mode = lambda path: int(oct(os.stat(path).st_mode & 0o777)[2:])
    expected_mode = chmod if chmod is not None else get_human_mode(spath)

    if mac_or_linux:
        assert get_human_mode(dpath) == expected_mode


@pytest.mark.parametrize('different', [True, False])
@pytest.mark.parametrize('raise_if_comparison_fails', [True, False])
def test_compare_directories(tmpdir_factory, different, raise_if_comparison_fails):
    hasher_cls = hashlib.sha256

    base_dir = str(tmpdir_factory.mktemp('ecephys_copy_utility_test_compare_directories'))
    sdir = os.path.join(base_dir, 'src')
    os.makedirs(sdir)
    ddir = os.path.join(base_dir, 'dest')
    os.makedirs(ddir)

    if different:

        with open(os.path.join(sdir, 'foo.txt'), 'w') as f:
            f.write('baz')

        if raise_if_comparison_fails:
            with pytest.raises(ValueError):
                cu.compare_directories(sdir, ddir, hasher_cls, raise_if_comparison_fails)
        else:
            with pytest.warns(UserWarning):
                cu.compare_directories(sdir, ddir, hasher_cls, raise_if_comparison_fails)

    else:
        cu.compare_directories(sdir, ddir, hasher_cls, raise_if_comparison_fails)


@pytest.mark.parametrize('different', [True, False])
@pytest.mark.parametrize('raise_if_comparison_fails', [True, False])
def test_compare_files(tmpdir_factory, different, raise_if_comparison_fails):
    hasher_cls = hashlib.sha256

    base_dir = str(tmpdir_factory.mktemp('ecephys_copy_utility_test_compare_files'))
    spath = os.path.join(base_dir, 'source.txt')
    dpath = os.path.join(base_dir, 'dest.txt')

    with open(spath, 'w') as f:
        f.write('baz')

    if different:

        with open(dpath, 'w') as f:
            f.write('fish')

        if raise_if_comparison_fails:
            with pytest.raises(ValueError):
                cu.compare_files(spath, dpath, hasher_cls, raise_if_comparison_fails)
        else:
            with pytest.warns(UserWarning):
                cu.compare_files(spath, dpath, hasher_cls, raise_if_comparison_fails)

    else:

        with open(dpath, 'w') as f:
            f.write('baz')

        cu.compare_files(spath, dpath, hasher_cls, raise_if_comparison_fails)
