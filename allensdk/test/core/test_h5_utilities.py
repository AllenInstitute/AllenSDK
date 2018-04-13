import functools

import h5py
import pytest
import numpy as np

import allensdk.core.h5_utilities as h5_utilities


@pytest.fixture
def mem_h5(request):
    my_file = h5py.File('my_file.h5', driver='core', backing_store=False)

    def fin():
        my_file.close()
    request.addfinalizer(fin)

    return my_file


@pytest.fixture
def simple_h5(mem_h5):
    mem_h5.create_group('a')
    mem_h5.create_group('a/b')
    mem_h5.create_group('a/b/c')
    mem_h5.create_group('d')
    mem_h5.create_group('a/e')

    return mem_h5


@pytest.fixture
def simple_h5_with_datsets(simple_h5):
    simple_h5.create_dataset(name='/a/b/c/fish', data=np.eye(10))
    simple_h5.create_dataset(name='a/fowl', data=np.eye(15))
    simple_h5.create_dataset(name='a/b/mammal', data=np.eye(20))

    return simple_h5


def test_decode_bytes():

    inp = np.array([b'a', b'b', b'c'])
    obt = h5_utilities.decode_bytes(inp)

    assert(np.array_equal( obt, ['a', 'b', 'c'] ))


def test_traverse_h5_file(simple_h5):

    names = []
    def cb(name, node):
        names.append(name)
    h5_utilities.traverse_h5_file(cb, simple_h5)

    assert( set(names) == set(['a', 'a/b', 'a/b/c', 'd', 'a/e']) )


def test_locate_h5_objects(simple_h5):

    matcher_cb = functools.partial(h5_utilities.h5_object_matcher_relname_in, ['c', 'e'])
    matches = h5_utilities.locate_h5_objects(matcher_cb, simple_h5)

    match_names = [ match.name for match in matches ]
    assert( set(match_names) == set(['/a/e', '/a/b/c']) )


def test_keyed_locate_h5_objects(simple_h5):

    matcher_cbs = {
        'e': functools.partial(h5_utilities.h5_object_matcher_relname_in, ['e']),
        'c': functools.partial(h5_utilities.h5_object_matcher_relname_in, ['c']),
    }

    matches = h5_utilities.keyed_locate_h5_objects(matcher_cbs, simple_h5)
    assert( matches['e'].name == '/a/e' )
    assert( matches['c'].name == '/a/b/c' )


def test_load_datasets_by_relnames(simple_h5_with_datsets):

    relnames = ['fish', 'fowl', 'mammal']
    obt = h5_utilities.load_datasets_by_relnames(relnames, simple_h5_with_datsets, simple_h5_with_datsets['a/b'])

    assert( len(obt) == 2 )
    assert(np.allclose( obt['fish'], np.eye(10) ))
    assert(np.allclose( obt['mammal'], np.eye(20) ))
