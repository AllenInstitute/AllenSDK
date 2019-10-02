import pytest
import mock
import numpy as np

from allensdk.internal.mouse_connectivity.tissuecyte_stitching.tile import Tile



@pytest.fixture(scope='function')
def small_tile():

    index = 20
    image = np.arange(200).reshape((10, 20)) # columns fast
    is_missing = False
    bounds = {'row': {'start': 40, 'end': 48}, 'column': {'start': 500, 'end': 516}}
    channel = 2
    size = {'row': 8, 'column': 16}
    margins = {'row': 1, 'column': 2}

    return Tile(index, image, is_missing, bounds, channel, size, margins)


def test_trim_self(small_tile):
    
    small_tile.trim_self()

    assert( np.allclose( small_tile.image.shape, [8, 16] ) )
    assert( np.amin(small_tile.image) == 22 )



def test_trim(small_tile):

    image = np.diag(np.arange(20))
    out = small_tile.trim(image)

    assert( np.allclose( out.shape, [8, 16] ) )
    assert( np.amax(out) == 8 )
    

@pytest.mark.parametrize('rs,cs,yn', [(8, 16, False), (11, 21, True)])
def test_average_tile_is_untrimmed(small_tile, rs, cs, yn):

    image = np.zeros((rs, cs))
    res = small_tile.average_tile_is_untrimmed(image)

    assert( res == yn )


@pytest.mark.parametrize('avt,do_trim', [(np.ones((8, 16)) * 2, True), 
                                         (np.ones((8, 16)) * 2, True), 
                                         (np.ones((10, 20)) * 2, True), 
                                         (np.ones((10, 20)) * 2, False)])
def test_apply_average_tile(small_tile, avt, do_trim):

    if do_trim:
        small_tile.trim_self()

    res = small_tile.apply_average_tile(avt)
    assert( np.allclose(res, small_tile.image * 2) )


def test_no_average_tile(small_tile):

    res = small_tile.apply_average_tile(None)
    assert( np.allclose(res, small_tile.image) )


def test_apply_average_tile_to_self(small_tile):

    av = np.ones((10, 20)) * 3
    prev = small_tile.image.copy()
    small_tile.apply_average_tile_to_self(av)

    assert( np.allclose( small_tile.image, prev * 3 ) )


def test_get_image_region(small_tile):

    image = np.zeros((1000, 1000, 3))
    image[:, :, 0] += 1
    image[:, :, 1] += 2
    image[:, :, 2] += 3
    image[45, 505, 2] = 4

    slc = small_tile.get_image_region()
    image = image[slc]
  
    assert( np.allclose( image.shape, [8, 16] ) )
    assert( image.sum() == 8 * 16 * 3 + 1 )


def test_get_missing_path(small_tile):

    obt = small_tile.get_missing_path()
    exp = [40, 500, 48, 500, 48, 516, 40, 516]

    assert( np.allclose(obt, exp) )


def test_initialize_image(small_tile):

    exp = np.zeros((8, 16))
    small_tile.initialize_image()

    assert( np.allclose( small_tile.image, exp ) )
