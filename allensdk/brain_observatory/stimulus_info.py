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

import six
import matplotlib.colors as mcolors
import numpy as np
import scipy.ndimage.interpolation as spndi
from scipy.misc import imresize
import matplotlib.pyplot as plt
from allensdk.api.cache import memoize
import itertools

# some handles for stimulus types
DRIFTING_GRATINGS = 'drifting_gratings'
DRIFTING_GRATINGS_SHORT = 'dg'
DRIFTING_GRATINGS_COLOR = '#a6cee3'

STATIC_GRATINGS = 'static_gratings'
STATIC_GRATINGS_SHORT = 'sg'
STATIC_GRATINGS_COLOR = '#1f78b4'

NATURAL_MOVIE_ONE = 'natural_movie_one'
NATURAL_MOVIE_ONE_SHORT = 'nm1'
NATURAL_MOVIE_ONE_COLOR = '#b2df8a'

NATURAL_MOVIE_TWO = 'natural_movie_two'
NATURAL_MOVIE_TWO_SHORT = 'nm2'
NATURAL_MOVIE_TWO_COLOR = '#33a02c'

NATURAL_MOVIE_THREE = 'natural_movie_three'
NATURAL_MOVIE_THREE_SHORT = 'nm3'
NATURAL_MOVIE_THREE_COLOR = '#fb9a99'

NATURAL_SCENES = 'natural_scenes'
NATURAL_SCENES_SHORT = 'ns'
NATURAL_SCENES_COLOR = '#e31a1c'

LOCALLY_SPARSE_NOISE = 'locally_sparse_noise'
LOCALLY_SPARSE_NOISE_SHORT = 'lsn'
LOCALLY_SPARSE_NOISE_COLOR = '#fdbf6f'

LOCALLY_SPARSE_NOISE_4DEG = 'locally_sparse_noise_4deg'
LOCALLY_SPARSE_NOISE_4DEG_SHORT = 'lsn4'
LOCALLY_SPARSE_NOISE_4DEG_COLOR = '#fdbf6f'

LOCALLY_SPARSE_NOISE_8DEG = 'locally_sparse_noise_8deg'
LOCALLY_SPARSE_NOISE_8DEG_SHORT = 'lsn8'
LOCALLY_SPARSE_NOISE_8DEG_COLOR = '#ff7f00'

SPONTANEOUS_ACTIVITY = 'spontaneous'
SPONTANEOUS_ACTIVITY_SHORT = 'sp'
SPONTANEOUS_ACTIVITY_COLOR = '#cab2d6'

# handles for stimulus names
THREE_SESSION_A = 'three_session_A'
THREE_SESSION_B = 'three_session_B'
THREE_SESSION_C = 'three_session_C'
THREE_SESSION_C2 = 'three_session_C2'

SESSION_STIMULUS_MAP = {
    THREE_SESSION_A: [DRIFTING_GRATINGS, NATURAL_MOVIE_ONE, NATURAL_MOVIE_THREE, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_B: [STATIC_GRATINGS, NATURAL_SCENES, NATURAL_MOVIE_ONE, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_C: [LOCALLY_SPARSE_NOISE, NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_C2: [LOCALLY_SPARSE_NOISE_4DEG, LOCALLY_SPARSE_NOISE_8DEG, NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, SPONTANEOUS_ACTIVITY]
}

LOCALLY_SPARSE_NOISE_STIMULUS_TYPES = [LOCALLY_SPARSE_NOISE, LOCALLY_SPARSE_NOISE_4DEG, LOCALLY_SPARSE_NOISE_8DEG]
NATURAL_MOVIE_STIMULUS_TYPES = [NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, NATURAL_MOVIE_THREE]

LOCALLY_SPARSE_NOISE_DIMENSIONS = {
    LOCALLY_SPARSE_NOISE: [ 16, 28 ],
    LOCALLY_SPARSE_NOISE_4DEG: [ 16, 28 ],
    LOCALLY_SPARSE_NOISE_8DEG: [ 8, 14 ],
    }

LOCALLY_SPARSE_NOISE_PIXELS = {
    LOCALLY_SPARSE_NOISE: 45,
    LOCALLY_SPARSE_NOISE_4DEG: 45,
    LOCALLY_SPARSE_NOISE_8DEG: 90,
    }

NATURAL_SCENES_PIXELS = (918, 1174)
NATURAL_MOVIE_PIXELS = (1080, 1920)
NATURAL_MOVIE_DIMENSIONS = (304, 608)

MONITOR_DIMENSIONS = (1200, 1920)
MONITOR_DISTANCE = 15

STIMULUS_GRAY = 127
STIMULUS_BITDEPTH = 8

def sessions_with_stimulus(stimulus):
    """ Return the names of the sessions that contain a given stimulus. """
    
    sessions = set()
    for session, session_stimuli in six.iteritems(SESSION_STIMULUS_MAP):
        if stimulus in session_stimuli:
            sessions.add(session)

    return sorted(list(sessions))


def stimuli_in_session(session):
    """ Return a list what stimuli are available in a given session.

    Parameters
    ----------
    session: string
        Must be one of: [stimulus_info.THREE_SESSION_A, stimulus_info.THREE_SESSION_B, stimulus_info.THREE_SESSION_C]
    """
    return SESSION_STIMULUS_MAP[session]


def all_stimuli():
    """ Return a list of all stimuli in the data set """
    return set([v for k, vl in six.iteritems(SESSION_STIMULUS_MAP) for v in vl])

class BinaryIntervalSearchTree(object):

    @staticmethod
    def from_df(input_df):
        search_list = input_df.to_dict('records')



        new_list = []
        for x in search_list:
            if x['start'] == x['end']:
               new_list.append((x['start'], x['end'], x))
            else:
               # -.01 prevents endpoint-overlapping intervals; assigns ties to intervals that start at requested index
               new_list.append((x['start'], x['end'] - .01, x))
        return BinaryIntervalSearchTree(new_list)


    def __init__(self, search_list):
        """Create a binary tree to search for a point within a list of intervals.  Assumes that the intervals are
        non-overlapping.  If two intervals share an endpoint, the left-side wins the tie.

        :param search_list: list of interval tuples; in the tuple, first element is interval start, then interval
        end (inclusive), then the return value for the lookup

        Example:
        bist = BinaryIntervalSearchTree([(0,.5,'A'), (1,2,'B')])
        print bist.search(1.5)
        """

        # Double-check that the list is sorted
        search_list = sorted(search_list, key=lambda x:x[0])

        # Check that the intervals are non-overlapping (except potentially at the end point)
        for x, y in zip(search_list[:-1], search_list[1:]):
            assert x[1] < y[0]


        self.data = {}
        self.add(search_list)

    def add(self, input_list, tmp=None):
        if tmp is None:
            tmp = []

        if len(input_list) == 1:
            self.data[tuple(tmp)] = input_list[0]
        else:
            self.add(input_list[:int(len(input_list)/2)], tmp=tmp+[0])
            self.add(input_list[int(len(input_list)/2):], tmp=tmp+[1])
            self.data[tuple(tmp)] = input_list[int(len(input_list)/2)-1]

    def search(self, fi, tmp=None):
        if tmp is None:
            tmp = []

        if (self.data[tuple(tmp)][0] <= fi) and (fi <= self.data[tuple(tmp)][1]):
            return_val = self.data[tuple(tmp)]
        elif fi < self.data[tuple(tmp)][1]:
            return_val = self.search(fi, tmp=tmp + [0])
        else:
            return_val = self.search(fi, tmp=tmp + [1])

        # print 'CHECKING:', return_val[0], fi, return_val[1], tmp
        assert (return_val[0] <= fi) and (fi <= return_val[1])
        return return_val

class StimulusSearch(object):

    def __init__(self, nwb_dataset):

        self.nwb_data = nwb_dataset
        self.epoch_df = nwb_dataset.get_stimulus_epoch_table()
        self.master_df = nwb_dataset.get_stimulus_table('master')
        self.epoch_bst = BinaryIntervalSearchTree.from_df(self.epoch_df)
        self.master_bst = BinaryIntervalSearchTree.from_df(self.master_df)

    @memoize
    def search(self, fi):

        try:

            # Look in fine-grain tree:
            search_result = self.master_bst.search(fi)
            return search_result
        except KeyError:

            # Current frame not found in a fine-grain interval;
            #   see if it is unregistered to a coarse-grain epoch:
            try:

                # THis will thow KeyError if not in coarse-grain epoch
                self.epoch_bst.search(fi)

                # Frame is in a coarse-grain  epoch, but not a fine grain interval;
                #   look backwards to find most recent find nearest matching interval
                if fi < self.epoch_df.iloc[0]['start']:

                    # Breakout if we go before the experiment:
                    return None
                else:
                    return self.search(fi-1)

            except KeyError:

                # Frame is unregistered at the coarse level; return None
                return None

def rotate(X, Y, theta):
    Xp = X * np.cos(theta) - Y * np.sin(theta)
    Yp = X * np.sin(theta) + Y * np.cos(theta)
    return Xp, Yp

def get_spatial_grating(height=None, aspect_ratio=None, ori=None, pix_per_cycle=None, phase=None, p2p_amp=2, baseline=0):

    aspect_ratio = float(aspect_ratio)
    _height_prime = 100

    sf = 1./(float(pix_per_cycle)/(height/float(_height_prime)))

    # Final height set by zoom below:
    y, x = (_height_prime,_height_prime*aspect_ratio)

    theta = ori * np.pi / 180.0  # convert to radians

    ph = phase * np.pi * 2.0

    X, Y = np.meshgrid(np.arange(x), np.arange(y))
    X = X - x / 2
    Y = Y - y / 2
    Xp, Yp = rotate(X, Y, theta)

    img = np.cos(2.0 * np.pi * Xp * sf + ph)
    return (p2p_amp/2.)*spndi.zoom(img, height/float(_height_prime)) + baseline

def get_spatio_temporal_grating(t, temporal_frequency=None, **kwargs):

    kwargs['phase'] = kwargs.pop('phase', 0) + (float(t)*temporal_frequency)%1

    return get_spatial_grating(**kwargs)

def map_template_coordinate_to_monitor_coordinate(template_coord, monitor_shape, template_shape):

    rx, cx = template_coord
    n_pixels_r, n_pixels_c = monitor_shape
    tr, tc = template_shape

    rx_new = float((n_pixels_r - tr) / 2) + rx
    cx_new = float((n_pixels_c - tc) / 2) + cx

    return rx_new, cx_new

def map_monitor_coordinate_to_template_coordinate(monitor_coord, monitor_shape, template_shape):

    rx, cx = monitor_coord
    n_pixels_r, n_pixels_c = monitor_shape
    tr, tc = template_shape

    rx_new =  rx - float((n_pixels_r - tr) / 2)
    cx_new =  cx - float((n_pixels_c - tc) / 2)

    return rx_new, cx_new

def lsn_coordinate_to_monitor_coordinate(lsn_coordinate, monitor_shape, stimulus_type):

    template_shape = LOCALLY_SPARSE_NOISE_DIMENSIONS[stimulus_type]
    pixels_per_patch = LOCALLY_SPARSE_NOISE_PIXELS[stimulus_type]

    rx, cx = lsn_coordinate
    tr, tc = template_shape

    return map_template_coordinate_to_monitor_coordinate((rx*pixels_per_patch, cx*pixels_per_patch),
                                                      monitor_shape,
                                                      (tr*pixels_per_patch, tc*pixels_per_patch))

def monitor_coordinate_to_lsn_coordinate(monitor_coordinate, monitor_shape, stimulus_type):

    pixels_per_patch = LOCALLY_SPARSE_NOISE_PIXELS[stimulus_type]
    tr, tc = LOCALLY_SPARSE_NOISE_DIMENSIONS[stimulus_type]

    rx, cx = map_monitor_coordinate_to_template_coordinate(monitor_coordinate, monitor_shape, (tr*pixels_per_patch, tc*pixels_per_patch))

    return (rx/pixels_per_patch, cx/pixels_per_patch)

def natural_scene_coordinate_to_monitor_coordinate(natural_scene_coordinate, monitor_shape):

    return map_template_coordinate_to_monitor_coordinate(natural_scene_coordinate, monitor_shape, NATURAL_SCENES_PIXELS)

def natural_movie_coordinate_to_monitor_coordinate(natural_movie_coordinate, monitor_shape):

    local_y = 1.*NATURAL_MOVIE_PIXELS[0]*natural_movie_coordinate[0]/NATURAL_MOVIE_DIMENSIONS[0]
    local_x = 1. * NATURAL_MOVIE_PIXELS[1] * natural_movie_coordinate[1] / NATURAL_MOVIE_DIMENSIONS[1]

    return map_template_coordinate_to_monitor_coordinate((local_y, local_x), monitor_shape, NATURAL_MOVIE_PIXELS)

def map_stimulus_coordinate_to_monitor_coordinate(template_coordinate, monitor_shape, stimulus_type):

    if stimulus_type in LOCALLY_SPARSE_NOISE_STIMULUS_TYPES:
        return lsn_coordinate_to_monitor_coordinate(template_coordinate, monitor_shape, stimulus_type)
    elif stimulus_type in NATURAL_MOVIE_STIMULUS_TYPES:
        return natural_movie_coordinate_to_monitor_coordinate(template_coordinate, monitor_shape)
    elif stimulus_type == NATURAL_SCENES:
        return natural_scene_coordinate_to_monitor_coordinate(template_coordinate, monitor_shape)
    elif stimulus_type in [DRIFTING_GRATINGS, STATIC_GRATINGS, SPONTANEOUS_ACTIVITY]:
        return template_coordinate
    else:
        raise NotImplementedError

def monitor_coordinate_to_natural_movie_coordinate(monitor_coordinate, monitor_shape):

    local_y, local_x = map_monitor_coordinate_to_template_coordinate(monitor_coordinate, monitor_shape, NATURAL_MOVIE_PIXELS)

    return float(NATURAL_MOVIE_DIMENSIONS[0])*local_y/NATURAL_MOVIE_PIXELS[0], float(NATURAL_MOVIE_DIMENSIONS[1])*local_x/NATURAL_MOVIE_PIXELS[1]

def map_monitor_coordinate_to_stimulus_coordinate(monitor_coordinate, monitor_shape, stimulus_type):

    if stimulus_type in LOCALLY_SPARSE_NOISE_STIMULUS_TYPES:
        return monitor_coordinate_to_lsn_coordinate(monitor_coordinate, monitor_shape, stimulus_type)
    elif stimulus_type == NATURAL_SCENES:
        return map_monitor_coordinate_to_template_coordinate(monitor_coordinate, monitor_shape, NATURAL_SCENES_PIXELS)
    elif stimulus_type in NATURAL_MOVIE_STIMULUS_TYPES:
        return monitor_coordinate_to_natural_movie_coordinate(monitor_coordinate, monitor_shape)
    elif stimulus_type in [DRIFTING_GRATINGS, STATIC_GRATINGS, SPONTANEOUS_ACTIVITY]:
        return monitor_coordinate
    else:
        raise NotImplementedError

def map_stimulus(source_stimulus_coordinate, source_stimulus_type, target_stimulus_type, monitor_shape):
    mc = map_stimulus_coordinate_to_monitor_coordinate(source_stimulus_coordinate, monitor_shape, source_stimulus_type)
    return map_monitor_coordinate_to_stimulus_coordinate(mc, monitor_shape, target_stimulus_type)

class Monitor(object):

    def __init__(self, n_pixels_r, n_pixels_c, panel_size, spatial_unit):

        self.spatial_unit = spatial_unit
        if spatial_unit == 'cm':
            self.spatial_conversion_factor = 1.
        else:
            raise NotImplementedError

        self._panel_size = panel_size
        self.n_pixels_r = n_pixels_r
        self.n_pixels_c = n_pixels_c

    @property
    def panel_size(self):
        return self._panel_size*self.spatial_conversion_factor

    @property
    def aspect_ratio(self):
        return float(self.n_pixels_c)/self.n_pixels_r

    @property
    def height(self):
        return self.spatial_conversion_factor*np.sqrt(self.panel_size**2/(1+self.aspect_ratio**2))

    @property
    def width(self):
        return self.height*self.aspect_ratio

    def set_spatial_unit(self, new_unit):
        if new_unit == self.spatial_unit:
            pass
        elif new_unit == 'inch' and self.spatial_unit == 'cm':
            self.spatial_conversion_factor *= .393701
        elif new_unit == 'cm' and self.spatial_unit == 'inch':
            self.spatial_conversion_factor *= 1./.393701
        else:
            raise NotImplementedError
        self.spatial_unit = new_unit

    @property
    def pixel_size(self):
        return float(self.width)/self.n_pixels_c

    def pixels_to_visual_degrees(self, n, distance_from_monitor, small_angle_approximation=True):

        if small_angle_approximation == True:
            return n*self.pixel_size/distance_from_monitor*57.2958 # radians to degrees
        else:
            return 2*np.arctan(n*1./2*self.pixel_size / distance_from_monitor) * 57.2958  # radians to degrees

    def lsn_image_to_screen(self, img, stimulus_type, origin='lower', background_color=STIMULUS_GRAY):

        assert img.dtype == np.uint8

        pixels_per_patch = LOCALLY_SPARSE_NOISE_PIXELS[stimulus_type]

        full_image = np.full((self.n_pixels_r, self.n_pixels_c), background_color, dtype=np.uint8)
        img_full_res = imresize(img, float(pixels_per_patch), interp='nearest')
        mr, mc = lsn_coordinate_to_monitor_coordinate((0, 0), (self.n_pixels_r, self.n_pixels_c), stimulus_type)
        Mr, Mc = lsn_coordinate_to_monitor_coordinate(img.shape, (self.n_pixels_r, self.n_pixels_c), stimulus_type)
        full_image[int(mr):int(Mr), int(mc):int(Mc)] = img_full_res

        if origin == 'lower':
            return full_image
        elif origin == 'upper':
            return np.flipud(full_image)
        else:
            raise Exception

        return full_image

    def natural_scene_image_to_screen(self, img, origin='lower'):

        # assert img.dtype == np.float32
        # img = img.astype(np.uint8)

        full_image = np.full((self.n_pixels_r, self.n_pixels_c), 127, dtype=np.uint8)
        mr, mc = natural_scene_coordinate_to_monitor_coordinate((0, 0), (self.n_pixels_r, self.n_pixels_c))
        Mr, Mc = natural_scene_coordinate_to_monitor_coordinate((img.shape[0], img.shape[1]), (self.n_pixels_r, self.n_pixels_c))
        full_image[int(mr):int(Mr), int(mc):int(Mc)] = img
        if origin == 'lower':
            return np.flipud(full_image)
        elif origin == 'upper':
            return full_image
        else:
            raise Exception

    def natural_movie_image_to_screen(self, img, origin='lower'):

        img = imresize(img, NATURAL_MOVIE_PIXELS)

        assert img.dtype == np.uint8

        full_image = np.full((self.n_pixels_r, self.n_pixels_c), 127, dtype=np.uint8)
        mr, mc = map_template_coordinate_to_monitor_coordinate((0, 0), (self.n_pixels_r, self.n_pixels_c), NATURAL_MOVIE_PIXELS)
        Mr, Mc = map_template_coordinate_to_monitor_coordinate((img.shape[0], img.shape[1]), (self.n_pixels_r, self.n_pixels_c), NATURAL_MOVIE_PIXELS)

        full_image[int(mr):int(Mr), int(mc):int(Mc)] = img

        if origin == 'lower':
            return np.flipud(full_image)
        elif origin == 'upper':
            return full_image
        else:
            raise Exception

    def spatial_frequency_to_pix_per_cycle(self, spatial_frequency, distance_from_monitor):

        # How many cycles do I want to see post warp:
        number_of_cycles = spatial_frequency*2*np.degrees(np.arctan(self.width/2./distance_from_monitor))

        # How many pixels to I have pre-warp to place my cycles on:
        mask = self.get_mask()
        _, m_col = np.where(mask != 0)
        number_of_pixels = (m_col.max() - m_col.min())

        return float(number_of_pixels)/number_of_cycles


    def grating_to_screen(self, phase, spatial_frequency, orientation, distance_from_monitor, p2p_amp=256, baseline=127):

        pix_per_cycle = self.spatial_frequency_to_pix_per_cycle(spatial_frequency, distance_from_monitor)

        grating = get_spatial_grating(height=self.n_pixels_r,
                                      aspect_ratio=self.aspect_ratio,
                                      ori=orientation,
                                      pix_per_cycle=pix_per_cycle,
                                      phase=phase,
                                      p2p_amp=p2p_amp,
                                      baseline=baseline)

        return grating

    def get_mask(self):

        from allensdk.core.brain_observatory_nwb_data_set import make_display_mask

        mask = make_display_mask(display_shape=(self.n_pixels_c, self.n_pixels_r)).T
        assert mask.shape[0] == self.n_pixels_r
        assert mask.shape[1] == self.n_pixels_c

        return mask

    def show_image(self, img, ax=None, show=True, mask=False, warp=False, origin='lower'):

        from allensdk.core.brain_observatory_nwb_data_set import make_display_mask

        assert img.shape == (self.n_pixels_r, self.n_pixels_c) or img.shape == (self.n_pixels_r, self.n_pixels_c, 4)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if warp == True:
            img = self.warp_image(img)

        if warp == True:
            assert mask == False

        ax.imshow(img, origin=origin, cmap=plt.cm.gray, interpolation='none')

        if mask == True:
            mask = make_display_mask(display_shape=(self.n_pixels_c, self.n_pixels_r)).T
            alpha_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            alpha_mask[:, :, 2] = 1 - mask
            alpha_mask[:, :, 3] = .4
            ax.imshow(alpha_mask, origin=origin, interpolation='none')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if origin == 'upper':
            ax.set_ylim((img.shape[0], 0))
        elif origin == 'lower':
            ax.set_ylim((0, img.shape[0]))
        else:
            raise Exception
        ax.set_xlim((0, img.shape[1]))

        if show == True:
            plt.show()

    def map_stimulus(self, source_stimulus_coordinate, source_stimulus_type, target_stimulus_type):
        monitor_shape = (self.n_pixels_r, self.n_pixels_c)
        return map_stimulus(source_stimulus_coordinate, source_stimulus_type, target_stimulus_type, monitor_shape)

class ExperimentGeometry(object):

    def __init__(self, distance, mon_height_cm, mon_width_cm, mon_res, eyepoint):

        self.distance = distance
        self.mon_height_cm = mon_height_cm
        self.mon_width_cm = mon_width_cm
        self.mon_res = mon_res
        self.eyepoint = eyepoint

        self._warp_coordinates = None

    @property
    def warp_coordinates(self):
        if self._warp_coordinates is None:
            self._warp_coordinates = self.generate_warp_coordinates()

        return self._warp_coordinates

    def generate_warp_coordinates(self):

        from allensdk.core.brain_observatory_nwb_data_set import warp_stimulus_coords

        display_shape=self.mon_res
        x = np.array(range(display_shape[0])) - display_shape[0] / 2
        y = np.array(range(display_shape[1])) - display_shape[1] / 2
        display_coords = np.array(list(itertools.product(y, x)))

        warp_coorinates = warp_stimulus_coords(display_coords,
                                               distance=self.distance,
                                               mon_height_cm=self.mon_height_cm,
                                               mon_width_cm=self.mon_width_cm,
                                               mon_res=self.mon_res,
                                               eyepoint=self.eyepoint)

        warp_coorinates[:, 0] += display_shape[1] / 2
        warp_coorinates[:, 1] += display_shape[0] / 2

        return warp_coorinates

class BrainObservatoryMonitor(Monitor):
    '''
    http://help.brain-map.org/display/observatory/Documentation?preview=/10616846/10813485/VisualCoding_VisualStimuli.pdf
    https://www.cnet.com/products/asus-pa248q/specs/
    '''

    def __init__(self, experiment_geometry=None):

        height, width = MONITOR_DIMENSIONS

        super(BrainObservatoryMonitor, self).__init__(height, width, 61.214, 'cm')

        if experiment_geometry is None:
            self.experiment_geometry = ExperimentGeometry(distance=float(MONITOR_DISTANCE), mon_height_cm=self.height, mon_width_cm=self.width, mon_res=(self.n_pixels_c, self.n_pixels_r), eyepoint=(0.5, 0.5))
        else:
            self.experiment_geometry = experiment_geometry

    def lsn_image_to_screen(self, img, **kwargs):

        if img.shape == tuple(LOCALLY_SPARSE_NOISE_DIMENSIONS[LOCALLY_SPARSE_NOISE]):
            return super(BrainObservatoryMonitor, self).lsn_image_to_screen(img, LOCALLY_SPARSE_NOISE, **kwargs)
        elif img.shape == tuple(LOCALLY_SPARSE_NOISE_DIMENSIONS[LOCALLY_SPARSE_NOISE_4DEG]):
            return super(BrainObservatoryMonitor, self).lsn_image_to_screen(img, LOCALLY_SPARSE_NOISE_4DEG, **kwargs)
        elif img.shape == tuple(LOCALLY_SPARSE_NOISE_DIMENSIONS[LOCALLY_SPARSE_NOISE_8DEG]):
            return super(BrainObservatoryMonitor, self).lsn_image_to_screen(img, LOCALLY_SPARSE_NOISE_8DEG, **kwargs)
        else:
            raise RuntimeError

    def warp_image(self, img, **kwargs):

        assert img.shape == (self.n_pixels_r, self.n_pixels_c)
        assert self.spatial_unit == 'cm'

        return spndi.map_coordinates(img, self.experiment_geometry.warp_coordinates.T).reshape((self.n_pixels_r, self.n_pixels_c))

    def grating_to_screen(self, phase, spatial_frequency, orientation):

        return super(BrainObservatoryMonitor, self).grating_to_screen(phase, spatial_frequency, orientation,
                                                                      self.experiment_geometry.distance,
                                                                      p2p_amp = 256, baseline = 127)

    def pixels_to_visual_degrees(self, n, **kwargs):

        return super(BrainObservatoryMonitor, self).pixels_to_visual_degrees(n, self.experiment_geometry.distance, **kwargs)