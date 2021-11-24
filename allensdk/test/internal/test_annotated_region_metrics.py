import pytest
import numpy as np
from allensdk.internal.brain_observatory import annotated_region_metrics

@pytest.fixture
def mask():
    return np.ones((10,10), dtype=bool)


def retinotopic_map(return_x=True):
    x = np.linspace(-np.pi, np.pi, 640)
    y = np.linspace(-np.pi, np.pi, 540)
    xx, yy = np.meshgrid(x, y)
    if return_x:
        return xx
    return yy


@pytest.fixture
def azimuth_map():
    return retinotopic_map()


@pytest.fixture
def altitude_map():
    return retinotopic_map(False)


def test_eccentricity(azimuth_map, altitude_map):
    ecc = annotated_region_metrics.eccentricity(azimuth_map, altitude_map,
                                                0.0, 0.0)
    assert(ecc.shape == azimuth_map.shape)


def test_create_region_mask(mask):
    height, width = mask.shape
    x = y = 30
    region_mask = annotated_region_metrics.create_region_mask((100,100),
                                                               x, y, width,
                                                               height,
                                                               mask.tolist())
    assert(region_mask.shape == (100,100))
    assert(region_mask.sum() == mask.sum())
    assert(np.all(region_mask[y:y+height,x:x+width] == mask))


def test_retinotopy_metric(azimuth_map, mask):
    height, width = mask.shape
    x = y = 30
    region_mask = annotated_region_metrics.create_region_mask(
        azimuth_map.shape, x, y, width, height, mask.tolist())
    rmin, rmax, rrange, rbias = annotated_region_metrics.retinotopy_metric(
        region_mask, azimuth_map)
    rmap = np.degrees(azimuth_map[np.where(region_mask > 0)])
    assert(rmin == rmap.min())
    assert(rmax == rmap.max())


def test_get_metrics(altitude_map, azimuth_map, mask):
    height, width = mask.shape
    x = y = 30
    result = annotated_region_metrics.get_metrics(altitude_map, azimuth_map,
                                                  x=x, y=y, width=width,
                                                  height=height,
                                                  mask=mask.tolist())
    assert(isinstance(result, dict))
    assert('azimuth_min' in result)
