import allensdk.brain_observatory.dff as dff
import numpy as np

def test_movingmode_fast():
    # check basic behavior
    x = np.array([0,10,0,0,20,0,0,0,30])
    kernelsize = 4
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(y == 0)

    # check window edges
    x = np.array([0,0,1,1,2,2,3,3])
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(x == y)

    # check > 16 bit
    x = np.array([4097, 4097, 4097, 4097])
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(y == 4097)

    # check floats
    x = np.array([0,0,1,1,2,2,3,3], dtype=np.float32)
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(x == y)

