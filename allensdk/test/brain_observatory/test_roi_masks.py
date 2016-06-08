import numpy as np
import allensdk.brain_observatory.roi_masks as roi_masks


def test_init_by_pixels():
    a = np.array([[0,0],[1,1],[1,0]])
    
    m = roi_masks.create_roi_mask(2, 2, [0,0,0,0], pix_list=a)
    
    mp = m.get_mask_plane()

    assert mp[0,0] == 1
    assert mp[1,1] == 1
    assert mp[1,0] == 0
    assert mp[1,1] == 1

    assert m.x == 0
    assert m.width == 2
    assert m.y == 0
    assert m.height == 2


def test_init_by_pixels_with_border():
    a = np.array([[1,1],[2,1]])
    
    m = roi_masks.create_roi_mask(3, 3, [1,1,1,1], pix_list=a)

    assert m.x == 1
    assert m.width == 2
    assert m.y == 1
    assert m.height == 1
    assert m.valid is False


def test_init_by_pixels_large():
    a = np.random.random((512,512))
    a[a > 0.5] = 1

    m = roi_masks.create_roi_mask(512, 512, [0,0,0,0], pix_list=np.argwhere(a))

    npx = len(np.where(a)[0])
    assert npx == len(np.where(m.get_mask_plane())[0])
