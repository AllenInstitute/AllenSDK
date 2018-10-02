import logging
import pandas as pd
import numpy as np
from pynwb import TimeSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from scipy.signal import medfilt
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.analyze import calc_deriv, rad_to_dist
from .timestamps import get_timestamps_from_sync

logger = logging.getLogger(__name__)


def visual_coding_running_speed(exp_data):
    # mostly copy-paste from visual_behavior runing speed code, but with different source
    dx_raw = exp_data['items']['foraging']['encoders'][0]['dx']
    time = exp_data['intervalsms']
    dx = medfilt(dx_raw, kernel_size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations

    if len(time) < len(dx):
        logger.error('intervalsms record appears to be missing entries')
        dx = dx[:len(time)]
        dx_raw = dx_raw[:len(time)]

    speed = calc_deriv(dx, time)
    speed = rad_to_dist(speed)

    running_speed = pd.DataFrame({
        'time': time,
        'frame': range(len(time)),
        'speed': speed,
        'dx': dx_raw,
    })
    return running_speed


class BaseStimulusAdapter(object):
    _source = ''

    def __init__(self, pkl_file, sync_file, stim_key='stim_vsync',
                 compress=True):
        self.pkl_file = pkl_file
        self.sync_file = sync_file
        self.stim_key = stim_key
        self._data = None
        if compress:
            self.compression_opts = {"compression": True,
                                     "compression_opts": 9}
        else:
            self.compression_opts = {}

    @property
    def core_data(self):
        raise NotImplementedError()

    def get_times(self):
        return get_timestamps_from_sync(self.sync_file, self.stim_key)

    @property
    def running_speed(self):
        running_df = self.core_data['running']
        speed = running_df.speed
        times = get_times()
        if len(times) > len(speed):
            logger.warning("Got times of length %s but speed of length %s, truncating times from the end",
                           len(times), len(speed))
            times = times[:len(speed)]

        ts = TimeSeries(name='running_speed',
                        source=self._source,
                        data=H5DataIO(speed.values, **self.compression_opts),
                        timestamps=times,
                        unit='cm/s')

        return ts


class VisualBehaviorStimulusAdapter(BaseStimulusAdapter):
    _source = 'Allen Brain Observatory: Visual Behavior'

    def __init__(self, pkl_file, sync_file=None, stim_key='stim_vsync', compress=True):
        '''Cleaning up init signature for optional kwarg sync'''

        super(VisualBehaviorStimulusAdapter, self).__init__(pkl_file, sync_file, stim_key='stim_vsync', compress=True)

    @property
    def core_data(self):
        if self._data is None:
            loaded = pd.read_pickle(self.pkl_file)
            self._data = data_to_change_detection_core(loaded)

        return self._data

    def get_times(self):
        if self.sync_file is not None:
            return super(VisualBehaviorStimulusAdapter, self).get_times()
        else:
            return self.core_data['time']


class VisualCodingStimulusAdapter(BaseStimulusAdapter):
    _source = 'Allen Brain Observatory: Visual Coding'

    @property
    def core_data(self):
        if self._data is None:
            loaded = pd.read_pickle(self.pkl_file)
            self._data = {'running': visual_coding_running_speed(loaded)}
        return self._data
