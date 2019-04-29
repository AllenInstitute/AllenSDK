import copy as cp

import pandas as pd
import numpy as np


class CamStimOnePickleStimFile(object):


    @property
    def stimuli(self):
        '''List of dictionaries containing information about individual stimuli
        '''
        return self.data['stimuli']


    @property
    def frames_per_second(self):
        '''Framerate of stimulus presentation
        '''
        return self.data['fps']


    @property
    def pre_blank_sec(self):
        '''Time (s) before initial stimulus presentation
        '''
        return self.data['pre_blank_sec']


    @property
    def angular_wheel_velocity(self):
        ''' Extract the mean angular velocity of the running wheel (degrees / s) for each 
        frame.
        '''
        return self.frames_per_second * self.wheel_rotation


    @property
    def wheel_rotation(self):
        ''' Extract the total rotation of the running wheel on each frame.
        '''

        try:
            result = self.data['items']['foraging']['encoders'][0]['dx']
        except (KeyError, IndexError):
            try:
                result = self.data['dx']
            except KeyError:
                raise KeyError('unable to extract angular running speed from this stimulus pickle')
                
        return np.array(result)            


    def __init__(self, data, **kwargs):
        self.data = data


    @classmethod
    def factory(cls, path, **kwargs):
        data = pd.read_pickle(path)
        return cls(data, **kwargs)