import datetime
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from .stimulus import VisualCodingStimulusAdapter
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO


class VisualCodingLegacyNwbAdapter(object):
    def __init__(self, nwb_one_file, compress=True):
        self._dataset = BrainObservatoryNwbDataSet(nwb_one_file)
        self._compress = compress

    @property
    def running_speed(self):
        dxcm, dxtime = self._dataset.get_running_speed()
        if self._compress:
            dxcm = H5DataIO(dxcm, compression=True, compression_opts=9)

        ts = TimeSeries(name='running_speed',
                        source='Allen Brain Observatory: Visual Coding',
                        data=dxcm,
                        timestamps=dxtime,
                        unit='cm/s')

        return ts


def write_nwb(output_file, stimulus_adapter, session_metadata):
    """Write Visual Coding NWB 2.0 file

    Parameters
    ----------
    output_file : string
        Name of the NWB file.
    stimulus_adapter
        Adapter providing pynwb representations of data derived from stimulus
        files.
    session_metadata : dict
        Dictionary of session metadata.
    """
    nwbfile = NWBFile(
        source='Allen Institute for Brain Science',
        session_description='Visual Coding Optical Physiology Session',
        identifier=session_metadata["session_id"],
        session_start_time=session_metadata["acquisition_date"],
        file_create_date=datetime.datetime.now()
    )

    nwbfile.add_acquisition(stimulus_adapter.running_speed)

    with NWBHDF5IO(output_file, mode='w') as io:
        io.write(nwbfile)
