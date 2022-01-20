from typing import List, Union

import pandas as pd
import numpy as np


class CamStimOnePickleStimFile(object):

    @property
    def data(self) -> dict:
        return self._data

    @property
    def presentation_intervals(self) -> np.ndarray:
        """Get an array of interval times (in ms) between stimulus
        presentations
        """
        return np.array(self._data["intervalsms"])

    @property
    def num_frames(self) -> int:
        """Get the number of stimulus frames presented during pkl session."""
        return len(self.presentation_intervals) + 1

    @property
    def stimuli(self):
        """List of dictionaries containing information about individual stimuli
        """
        return self._data["stimuli"]

    @property
    def frames_per_second(self):
        """Framerate of stimulus presentation
        """
        return self._data["fps"]

    @property
    def pre_blank_sec(self):
        """Time (s) before initial stimulus presentation
        """
        return self._data["pre_blank_sec"]

    @property
    def angular_wheel_velocity(self):
        """ Extract the mean angular velocity of the running wheel
        (degrees / s) for each frame.
        """
        return self.frames_per_second * self.angular_wheel_rotation

    @property
    def angular_wheel_rotation(self):
        """ Extract the total rotation of the running wheel on each frame.
        """
        return self._extract_running_array("dx")

    @property
    def vsig(self):
        """Running speed signal voltage
        """
        return self._extract_running_array("vsig")

    @property
    def vin(self):
        return self._extract_running_array("vin")

    def __init__(self, data, **kwargs):
        self._data = data

    def _extract_running_array(self, key):
        try:
            result = self._data["items"]["foraging"]["encoders"][0][key]
        except (KeyError, IndexError):
            try:
                result = self._data[key]
            except KeyError:
                raise KeyError(
                    f"Unable to extract {key} from this stimulus pickle!"
                )

        return np.array(result)

    @classmethod
    def factory(cls, path, **kwargs):
        data = pd.read_pickle(path)
        return cls(data, **kwargs)


class BehaviorPickleFile(object):
    """A helper class to abstract common data/metadata lookup operations
    with a visual behavior session pkl file.
    """

    def _behavior_params(self) -> dict:
        """Shortcut to access behavior pkl behavior specific params"""
        return self._data["items"]["behavior"]

    @property
    def data(self) -> dict:
        """Pkl data in dictionary form"""
        return self._data

    @property
    def presentation_intervals(self) -> np.ndarray:
        """Get an array of interval times (in ms) between stimulus
        presentations
        """
        behavior = self._behavior_params()
        return np.array(behavior["intervalsms"])

    @property
    def image_set(self) -> str:
        """Get the name of the image set that was presented during the pkl
        session.

        Names will look something like:
        'Natural_Images_Lum_Matched_set_ophys_G_2019'
        """
        behavior = self._behavior_params()
        image_set_path = behavior["params"]["stimulus"]["params"]["image_set"]
        image_set = image_set_path.split('/')[-1].split('.')[0]
        return image_set

    @property
    def num_frames(self) -> int:
        """Get the number of stimulus frames presented during pkl session."""
        return len(self.presentation_intervals) + 1

    @property
    def reward_frames(self) -> np.ndarray:
        """Get the frames where a reward was delivered during pkl session."""
        behavior = self._behavior_params()
        reward_frames = behavior["rewards"][0]["reward_times"][:, 1]
        return reward_frames

    def __init__(self, data, **kwargs):
        self._data = data

    @classmethod
    def factory(cls, path, **kwargs) -> "BehaviorPickleFile":
        data = pd.read_pickle(path)
        return cls(data, **kwargs)


class ReplayPickleFile(object):

    @property
    def data(self) -> dict:
        """Pkl data in dictionary form"""
        return self._data

    @property
    def presentation_intervals(self) -> np.ndarray:
        """Get an array of interval times (in ms) between stimulus
        presentations
        """
        return np.array(self._data["intervalsms"])

    @property
    def num_frames(self) -> int:
        """Get the number of stimulus frames presented during pkl session."""
        return len(self.presentation_intervals) + 1

    @property
    def image_presentations(self) -> List[Union[str, None]]:
        """Get list of images presented during a replay session."""
        return self._data['stimuli'][0]['sweep_params']['ReplaceImage'][0]

    @property
    def unique_image_names(self) -> List[str]:
        """Get a list of unique images names presented during a replay session.
        """
        unique_image_names: set = set(self.image_presentations)
        unique_image_names.discard(None)
        return sorted(list(unique_image_names))

    def __init__(self, data, **kwargs):
        self._data = data

    @classmethod
    def factory(cls, path, **kwargs):
        data = pd.read_pickle(path)
        return cls(data, **kwargs)
