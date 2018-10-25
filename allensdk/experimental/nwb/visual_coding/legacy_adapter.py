from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from pynwb import TimeSeries


class VisualCodingLegacyNwbAdapter(object):
    def __init__(self, nwb_one_file):
        self._dataset = BrainObservatoryNwbDataSet(nwb_one_file)

    @property
    def running_speed(self):
        dxcm, dxtime = self._dataset.get_running_speed()

        ts = TimeSeries(name='running_speed',
                        source='Allen Brain Observatory: Visual Coding',
                        data=dxcm,
                        timestamps=dxtime,
                        unit='cm/s')

        return ts
