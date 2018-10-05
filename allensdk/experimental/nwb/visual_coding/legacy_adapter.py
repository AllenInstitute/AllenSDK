from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from pynwb import TimeSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO


class VisualCodingLegacyNwbAdapter(object):
    def __init__(self, nwb_one_file, compress=True):
        self._dataset = BrainObservatoryNwbDataSet(nwb_one_file)
        if compress:
            self.compression_opts = {"compression": True,
                                     "compression_opts": 9}
        else:
            self.compression_opts = {}

    @property
    def running_speed(self):
        dxcm, dxtime = self._dataset.get_running_speed()

        ts = TimeSeries(name='running_speed',
                        source='Allen Brain Observatory: Visual Coding',
                        data=H5DataIO(dxcm, **self.compression_opts),
                        timestamps=dxtime,
                        unit='cm/s')

        return ts
