# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import math
import scipy.ndimage.morphology as morphology
import logging

# constants used for accessing border array
RIGHT_SHIFT = 0
LEFT_SHIFT = 1
DOWN_SHIFT = 2
UP_SHIFT = 3


class Mask(object):
    '''
    Abstract class to represent image segmentation mask. Its two
    main subclasses are RoiMask and NeuropilMask. The former represents
    the mask of a region of interest (ROI), such as a cell observed in
    2-photon imaging. The latter represents the neuropil around that cell,
    and is useful when subtracting the neuropil signal from the measured
    ROI signal.

    This class should not be instantiated directly.

    Parameters
    ----------
    image_w: integer
       Width of image that ROI resides in

    image_h: integer
       Height of image that ROI resides in

    label: text
       User-defined text label to identify mask

    mask_group: integer
       User-defined number to help put masks into different categories
    '''

    def __init__(self, image_w, image_h, label, mask_group):
        '''
        Mask class constructor. The Mask class is designed to be abstract
        and it should not be instantiated directly.
        '''

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

    def init_by_pixels(self, border, pix_list):
        '''
        Initialize mask using a list of mask pixels

        Parameters
        ----------
        border: float[4]
            Coordinates defining useable area of image. See create_roi_mask()

        pix_list: integer[][2]
            List of pixel coordinates (x,y) that define the mask
        '''
        assert pix_list.shape[1] == 2, "Pixel list not properly formed"
        array = np.zeros((self.img_rows, self.img_cols))

        # pix_list stores array of [x,y] coordinates
        array[pix_list[:, 1], pix_list[:, 0]] = 1

        self.init_by_mask(border, array)

    def get_mask_plane(self):
        '''
        Returns mask content on full-size image plane

        Returns
        -------
        numpy 2D array [img_rows][img_cols]
        '''
        mask = np.zeros((self.img_rows, self.img_cols))
        mask[self.y:self.y + self.height, self.x:self.x + self.width] = self.mask
        return mask


def create_roi_mask(image_w, image_h, border, pix_list=None, roi_mask=None, label=None, mask_group=-1):
    '''
    Conveninece function to create and initializes an RoiMask

    Parameters
    ----------

    image_w: integer
        Width of image that ROI resides in

    image_h: integer
        Height of image that ROI resides in

    border: float[4]
        Coordinates defining useable area of image. If the entire image
        is usable, and masks are valid anywhere in the image, this should
        be [(image_w-1), 0, (image_h-1), 0]. The following constants
        help describe the array order:

            RIGHT_SHIFT = 0

            LEFT_SHIFT = 1

            DOWN_SHIFT = 2

            UP_SHIFT = 3

        When parts of the image are unusable, for example due motion
        correction shifting of different image frames, the border array
        should store the usable image area

    pix_list: integer[][2]
        List of pixel coordinates (x,y) that define the mask

    roi_mask: integer[image_h][image_w]
        Image-sized array that describes the mask. Active parts of the
        mask should have values >0. Background pixels must be zero

    label: text
        User-defined text label to identify mask

    mask_group: integer
        User-defined number to help put masks into different categories

    Returns
    -------
        RoiMask object
    '''
    m = RoiMask(image_w, image_h, label, mask_group)
    if pix_list is not None:
        m.init_by_pixels(border, pix_list)
    elif roi_mask is not None:
        m.init_by_mask(border, roi_mask)
    else:
        assert False, "Must specify either roi_mask or pix_list"
    return m


class RoiMask(Mask):

    def __init__(self, image_w, image_h, label, mask_group):
        '''
        RoiMask class constructor

        Parameters
        ----------
        image_w: integer
            Width of image that ROI resides in

        image_h: integer
            Height of image that ROI resides in

        label: text
            User-defined text label to identify mask

        mask_group: integer
            User-defined number to help put masks into different categories
        '''
        super(RoiMask, self).__init__(image_w, image_h, label, mask_group)

    def init_by_mask(self, border, array):
        '''
        Initialize mask using spatial mask

        Parameters
        ----------
        border: float[4]
            Coordinates defining useable area of image. See create_roi_mask().

        roi_mask: integer[image height][image width]
            Image-sized array that describes the mask. Active parts of the
            mask should have values >0. Background pixels must be zero
        '''
        # find lowest and highest non-zero indices on each axis
        px = np.argwhere(array)
        (top, left), (bottom, right) = px.min(0), px.max(0)

        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT]) - 1
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT]) - 1

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
        self.mask = array[top:bottom + 1, left:right + 1]


def create_neuropil_mask(roi, border, combined_binary_mask, label=None):
    '''
    Conveninece function to create and initializes a Neuropil mask.
    Neuropil masks are defined as the region around an ROI, up to 13
    pixels out, that does not include other ROIs

    Parameters
    ----------

    roi: RoiMask object
        The ROI that the neuropil masks will be based on

    border: float[4]
        Coordinates defining useable area of image. See create_roi_mask().

    combined_binary_mask
        List of pixel coordinates (x,y) that define the mask

    combined_binary_mask: integer[image_h][image_w]
        Image-sized array that shows the position of all ROIs in the
        image. ROI masks should have a value of one. Background pixels
        must be zero. In other words, ithe combined_binary_mask is a
        bitmap union of all ROI masks

    label: text
        User-defined text label to identify the mask

    Returns
    -------
        NeuropilMask object
    '''
    # combined_binary_mask is a bitmap union of ALL ROI masks
    # create a binary mask of the ROI
    binary_mask = np.zeros((roi.img_rows, roi.img_cols))
    binary_mask[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width] = roi.mask
    binary_mask = binary_mask > 0
    # dilate the mask
    binary_mask_dilated = morphology.binary_dilation(
        binary_mask, structure=np.ones((3, 3)), iterations=13)  # T/F
    # eliminate ROIs from the dilation
    binary_mask_dilated = binary_mask_dilated > combined_binary_mask
    # create mask from binary dilation
    m = NeuropilMask(w=roi.img_cols, h=roi.img_rows,
                     label=label, mask_group=roi.mask_group)
    m.init_by_mask(border, binary_mask_dilated)
    return m


class NeuropilMask(Mask):

    def __init__(self, w, h, label, mask_group):
        '''
        NeuropilMask class constructor. This class should be created by
        calling create_neuropil_mask()

        Parameters
        ----------
        label: text
            User-defined text label to identify mask

        mask_group: integer
            User-defined number to help put masks into different categories
        '''
        super(NeuropilMask, self).__init__(w, h, label, mask_group)

    def init_by_mask(self, border, array):
        '''
        Initialize mask using spatial mask

        Parameters
        ----------
        border: float[4]
            Coordinates defining useable area of image. See create_roi_mask().

        array: integer[image height][image width]
            Image-sized array that describes the mask. Active parts of the
            mask should have values >0. Background pixels must be zero
        '''
        # find lowest and highest non-zero indices on each axis
        px = np.argwhere(array)
        (top, left), (bottom, right) = px.min(0), px.max(0)

        # left and right border insets
        l_inset = math.ceil(border[RIGHT_SHIFT])
        r_inset = math.floor(self.img_cols - border[LEFT_SHIFT]) - 1
        # top and bottom border insets
        t_inset = math.ceil(border[DOWN_SHIFT])
        b_inset = math.floor(self.img_rows - border[UP_SHIFT]) - 1
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
            bottom = b_inset
            if top > b_inset:
                top = b_inset
        #
        self.x = left
        self.width = right - left + 1
        self.y = top
        self.height = bottom - top + 1
        # make copy of mask
        self.mask = array[top:bottom + 1, left:right + 1]


def calculate_traces(stack, mask_list):
    '''
    Calculates the average response of the specified masks in the
    image stack

    Parameters
    ----------
    stack: float[image height][image width]
        Image stack that masks are applied to

    mask_list: list<Mask>
        List of masks

    Returns
    -------
    float[number masks][number frames]
        This is the average response for each Mask in each image frame
    '''
    traces = np.zeros((len(mask_list), stack.shape[0]))
    num_frames = stack.shape[0]
    # make sure masks are numpy objects
    for mask in mask_list:
        if not isinstance(mask.mask, np.ndarray):
            mask.mask = np.array(mask.mask)
    # calculate traces
    for frame_num in range(num_frames):
        if frame_num % 1000 == 0:
            logging.debug("frame " + str(frame_num) + " of " + str(num_frames))
        frame = stack[frame_num]
        mask = None
        try:
            for i in range(len(mask_list)):
                mask = mask_list[i]
                subframe = frame[mask.y:mask.y +
                                 mask.height, mask.x:mask.x + mask.width]
                total = (subframe * mask.mask).sum(axis=-1).sum(axis=-1)
                area = (mask.mask).sum(axis=-1).sum(axis=-1)
                tvals = total / area
                traces[i][frame_num] = tvals
        except:
            logging.error("Error encountered processing mask during frame %d" % frame_num)
            if mask is not None:
                logging.error(subframe.shape)
                logging.error(mask.mask.shape)
                logging.error(mask)
            raise
    return traces


def create_roi_mask_array(rois):
    '''Create full image mask array from list of RoiMasks.

    Parameters
    ----------
    rois: list<RoiMask>
        List of roi masks.

    Returns
    -------
    np.ndarray: NxWxH array
        Boolean array of of len(rois) image masks.
    '''
    if rois:
        height = rois[0].img_rows
        width = rois[0].img_cols
        masks = np.zeros((len(rois), height, width), dtype=np.uint8)
        for i, roi in enumerate(rois):
            masks[i, :, :] = roi.get_mask_plane()
    else:
        masks = None
    return masks
