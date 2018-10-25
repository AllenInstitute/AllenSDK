import logging
import pandas as pd
import numpy as np
from pynwb import TimeSeries
from pynwb.epoch import Epochs
from scipy.signal import medfilt
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.analyze import compute_running_speed
from .timestamps import get_timestamps_from_sync
from pynwb.image import ImageSeries, IndexSeries
import numpy as np

logger = logging.getLogger(__name__)


def visual_coding_running_speed(exp_data):
    dx_raw = exp_data['items']['foraging']['encoders'][0]['dx']
    v_sig = exp_data['items']['foraging']['encoders'][0]['vsig']
    v_in = exp_data['items']['foraging']['encoders'][0]['vin']
    time = exp_data['intervalsms']

    if len(time) < len(dx_raw):
        logger.error('intervalsms record appears to be missing entries')
        dx_raw = dx_raw[:len(time)]
        v_sig = v_sig[:len(time)]
        v_in = v_in[:len(time)]

    speed = compute_running_speed(dx_raw, time, v_sig, v_in)

    running_speed = pd.DataFrame({
        'time': time,
        'frame': range(len(time)),
        'speed': speed,
        'dx': dx_raw,
    })
    return running_speed


class BaseStimulusAdapter(object):
    _source = ''

    def __init__(self, pkl_file, sync_file, stim_key='stim_vsync'):
        self.pkl_file = pkl_file
        self.sync_file = sync_file
        self.stim_key = stim_key
        self._data = None
        self._running_speed = None

        self.EPOCHS = 'epochs' # Dont change this without looking at https://github.com/NeurodataWithoutBorders/pynwb/issues/646


    @property
    def core_data(self):
        raise NotImplementedError()

    @property
    def session_start_time(self):
        raise NotImplementedError()

    @property
    def image_series_list(self):
        raise NotImplementedError()

    @property
    def index_series_list(self):
        raise NotImplementedError()

    def get_times(self):
        return get_timestamps_from_sync(self.sync_file, self.stim_key)

    @property
    def running_speed(self):

        if self._running_speed is None:

            running_df = self.core_data['running']
            speed = running_df.speed
            times = self.get_times()
            if len(times) > len(speed):
                logger.warning("Got times of length %s but speed of length %s, truncating times from the end",
                            len(times), len(speed))
                times = times[:len(speed)]

            self._running_speed = TimeSeries(name='running_speed',
                            source=self._source,
                            data=speed.values,
                            timestamps=times,
                            unit='cm/s')

        return self._running_speed


class VisualBehaviorStimulusAdapter(BaseStimulusAdapter):
    _source = 'Allen Brain Observatory: Visual Behavior'

    def __init__(self, pkl_file, sync_file=None, stim_key='stim_vsync'):
        '''Cleaning up init signature for optional kwarg sync'''

        super(VisualBehaviorStimulusAdapter, self).__init__(pkl_file, sync_file, stim_key=stim_key)

        self._stimulus_epoch_df = None
        self._stimulus_epoch_table = None
        self._visual_stimulus_image_series = None
        self._index_series_list = None

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

    @property
    def timestamp_source(self):
        if self.sync_file is not None:
            return 'sync_file_timestamps'
        else:
            return 'pkl_file_timestamps'

    @property
    def session_start_time(self):
        return self.core_data['metadata']['startdatetime']

    @property
    def stimulus_epoch_df(self):

        if self._stimulus_epoch_df is None:

            timestamps = self.get_times()
            df = self.core_data['visual_stimuli'].copy()
            df['stop_time'] = timestamps[df['end_frame']]
            df['start_time'] = timestamps[df['frame']]
            df['description'] = ['stimulus presentation']*len(df) 
            df['timeseries'] = [[self.running_speed]]*len(df) 
            df['tags'] = [[self.timestamp_source]]*len(df) 
            df.drop('time', inplace=True, axis=1)
            self._stimulus_epoch_df = df

        return self._stimulus_epoch_df

    @property
    def stimulus_epoch_table(self):

        if self._stimulus_epoch_table is None:
            self._stimulus_epoch_table = Epochs.from_dataframe(self.stimulus_epoch_df, 'nosource', self.EPOCHS)

        return self._stimulus_epoch_table

    @property
    def visual_stimulus_image_series(self):

        if self._visual_stimulus_image_series is None:

            image_set = self.core_data['image_set']
            name = image_set.get('name', 'TODO_visual_behavior_analysis_issues_389')
            image_data =  np.array(image_set['images'])
            source = image_set['metadata']['image_set']

            
            image_index = []
            for x in image_set['image_attributes']:
                image_index.append(x['image_index'])

            self._visual_stimulus_image_series = ImageSeries(
                                                name=name,
                                                source=source,
                                                data=image_data,
                                                unit='NA',
                                                format='raw',
                                                timestamps=image_index)

            

        return self._visual_stimulus_image_series


    @property
    def image_series_list(self):

        return [self.visual_stimulus_image_series]


    @property
    def index_series_list(self):
        
        if self._index_series_list is None:

            stimulus_epoch_df = self.stimulus_epoch_df
            image_set = self.core_data['image_set']

            mapper_dict = {}
            for x in image_set['image_attributes']:
                mapper_dict[x['image_name'], x['image_category']] = x['image_index']

            index_timeseries = []
            for cn, cc in zip(stimulus_epoch_df['image_name'].values, stimulus_epoch_df['image_category'].values):
                index_timeseries.append(mapper_dict[cn,cc])

            image_index_series = IndexSeries(
                            name='image_index',
                            source='NA',
                            data=index_timeseries,
                            unit='NA',
                            indexed_timeseries=self.visual_stimulus_image_series,
                            timestamps=stimulus_epoch_df['start_time'].values)

            self._index_series_list  = [image_index_series]
   
        return self._index_series_list 



class VisualCodingStimulusAdapter(BaseStimulusAdapter):
    _source = 'Allen Brain Observatory: Visual Coding'

    @property
    def core_data(self):
        if self._data is None:
            loaded = pd.read_pickle(self.pkl_file)
            self._data = {'running': visual_coding_running_speed(loaded)}
        return self._data
