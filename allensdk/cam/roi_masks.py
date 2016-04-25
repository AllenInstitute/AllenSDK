import numpy as np
import math
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology
from motion_border import *

# TODO document and format for SDK

class Mask(object):
    def __init__(self, image_w, image_h, label, mask_group):
        self.img_rows = image_h
        self.img_cols = image_w
        # initialize to invalid state. Mask must be manually initialized 
        #   by pixel list or mask array
        self.x = 0
        self.width = 0
        self.y = 0
        self.height = 0
        self.mask = None
        self.valid = True
        # label is for distinguishing neuropil from ROI, in case 
        #   these masks are mixed together
        self.label = label
        # auxiliary metadata. if a particula mask is part of an group,
        #   that data can be stored here
        self.mask_group = mask_group

    def __str__(self):
        return "%s: TL=%d,%d w,h=%d,%d\n%s" % (self.label, self.x, self.y, self.width, self.height, str(self.mask))


def create_roi_mask(image_w, image_h, border, pix_list=None, roi_mask=None, label=None, mask_group=-1):
    m = ROI_Mask(image_w, image_h, label, mask_group)
    if pix_list is not None:
        m.init_by_pixels(border, pix_list)
    elif roi_mask is not None:
        m.init_by_mask(border, roi_mask)
    else:
        assert False, "Must specify either roi_mask or pix_list"
    return m


class ROI_Mask(Mask):
    def __init__(self, image_w, image_h, label, mask_group):
        super(ROI_Mask, self).__init__(image_w, image_h, label, mask_group)

    def init_by_pixels(self, border, pix_list):
        assert pix_list.shape[1] == 2, "Pixel list not properly formed"
        array = np.zeros((self.img_rows, self.img_cols))
        # pix_list stores array of [x,y] coordinates
        for pix in pix_list:
            array[pix[1], pix[0]] = 1;
        self.init_by_mask(border, array)

    def init_by_mask(self, border, array):
        # find lowest and highest non-zero indices on each axis
        left = None
        right = None
        top = None
        bottom = None
        for r in range(self.img_rows):
            for c in range(self.img_cols):
                val = array[r][c]
                if val > 0:
                    if top is None or r < top:
                        top = r
                    if bottom is None or r > bottom:
                        bottom = r
                    if left is None or c < left:
                        left = c
                    if right is None or c > right:
                        right = c
        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT])
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT])
        # if ROI crosses border, it's considered invalid
        if left < l_inset or right > r_inset:
            self.valid = False
        if top < t_inset or bottom > b_inset:
            self.valid = False
        #
        self.x = left
        self.width = right - left + 1
        self.y = top
        self.height = bottom - top + 1
        # make copy of mask
        self.mask = array[top:bottom+1, left:right+1]


def create_neuropil_mask(roi, border, combined_binary_mask, label=None):
    # combined_binary_mask is a bitmap union of ALL ROI masks
    
    # create a binary mask of the ROI
    binary_mask = np.zeros((roi.img_rows, roi.img_cols))
    binary_mask[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width] = roi.mask
    binary_mask = binary_mask > 0
    # dilate the mask
    binary_mask_dilated = morphology.binary_dilation(binary_mask, structure=np.ones((3,3)), iterations=13)  # T/F
    # eliminate ROIs from the dilation
    binary_mask_dilated = binary_mask_dilated > combined_binary_mask
    # create mask from binary dilation
    m = Neuropil_Mask(w=roi.img_cols, h=roi.img_rows, label=label, mask_group=roi.mask_group)
    m.init_by_mask(border, binary_mask_dilated)
    return m


class Neuropil_Mask(Mask):
    def __init__(self, w, h, label, mask_group):
        super(Neuropil_Mask, self).__init__(w, h, label, mask_group)


    def init_by_pixels(self, border, pix_list):
        assert pix_list.shape[1] == 2, "Pixel list not properly formed"
        array = np.zeros((self.img_rows, self.img_cols))
        # pix_list stores array of [x,y] coordinates
        for pix in pix_list:
            array[pix[1], pix[0]] = 1;
        self.init_by_mask(border, array)


    def init_by_mask(self, border, array):
        # find lowest and highest non-zero indices on each axis
        left = None
        right = None
        top = None
        bottom = None
        for r in range(self.img_rows):
            for c in range(self.img_cols):
                val = array[r][c]
                if val > 0:
                    if left is None or c < left:
                        left = c
                    if right is None or c > right:
                        right = c
                    if top is None or r < top:
                        top = r
                    if bottom is None or r > bottom:
                        bottom = r
        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT])
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT])
        # restrict neuropil masks to center area of frame (ie, exclude 
        #   areas that overlap with movement correction buffer)
        if left < l_inset:
            left = l_inset
            if right < l_inset:
                right = l_inset
        if right > r_inset:
            right = r_inset
            if left > r_inset:
                left = r_inset
        if top < t_inset:
            top = t_inset
            if bottom < t_inset:
                bottom = t_inset
        if bottom > b_inset:
            bottom = b_inset;
            if top > b_inset:
                top = b_inset
        #
        self.x = left
        self.width = right - left + 1
        self.y = top
        self.height = bottom - top + 1
        # make copy of mask
        self.mask = array[top:bottom+1, left:right+1]


def calculate_traces(stack, mask_list):
    traces = np.zeros((len(mask_list), stack.shape[0]))
    num_frames = stack.shape[0]
    # make sure masks are numpy objects
    for mask in mask_list:
        if not isinstance(mask.mask, np.ndarray):
            mask.mask = np.array(mask.mask)
    # calcualte traces
    for frame_num in range(num_frames):
        if frame_num % 1000 == 0 :
            print "frame " + str(frame_num) + " of " + str(num_frames)
        frame = stack[frame_num]
        mask = None
        try:
            for i in range(len(mask_list)):
                    mask = mask_list[i]
                    subframe = frame[mask.y:mask.y+mask.height, mask.x:mask.x+mask.width]
                    total = (subframe * mask.mask).sum(axis=-1).sum(axis=-1)
                    area = (mask.mask).sum(axis=-1).sum(axis=-1)
                    if area == 0:
                        raise ValueError("Numerical error in mask %d" % i)

                    tvals = total/area
                    traces[i][frame_num] = tvals
        except:
            print("Error encountered processing mask during frame %d" % frame_num)
            if mask is not None:
                print subframe.shape
                print mask.mask.shape
                print mask
            raise
    return traces

