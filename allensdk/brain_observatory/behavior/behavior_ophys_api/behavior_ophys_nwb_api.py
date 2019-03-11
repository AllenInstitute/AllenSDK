import datetime
from pynwb import NWBFile, NWBHDF5IO
import pandas as pd
import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.running_speed import RunningSpeed


class BehaviorOphysNwbApi(object):

    @property
    def nwbfile(self):
        if hasattr(self, '_nwbfile'):
            return self._nwbfile

        io = NWBHDF5IO(self.path, 'r')
        return io.read()

    def __init__(self, path):
        self.path = path

    def save(self, session_object):

        nwbfile = NWBFile(
            session_description=str(session_object.metadata['session_type']),
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata['experiment_date'],
            file_create_date=datetime.datetime.now()
        )

        unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        nwb.add_running_data_df_to_nwbfile(nwbfile, session_object.running_data_df, unit_dict)

        with NWBHDF5IO(self.nwb_filepath, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_running_data_df(self, **kwargs):

        return pd.DataFrame({'v_sig': self.nwbfile.get_acquisition('v_sig').data,
                             'v_in': self.nwbfile.get_acquisition('v_in').data,
                             'speed': self.nwbfile.modules['running'].get_data_interface('speed').data,
                             'dx': self.nwbfile.modules['running'].get_data_interface('dx').data},
                                 index=pd.Index(self.nwbfile.modules['running'].get_data_interface('timestamps').timestamps, name='timestamps'))

    def get_metadata(self, **kwargs):
        pass

    def get_running_speed(self, **kwargs):
        
        running_data_df = self.get_running_data_df()
        return RunningSpeed(timestamps=running_data_df.index.values,
                            values=running_data_df.speed.values)


        # stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        # behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        # data = pd.read_pickle(behavior_stimulus_file)
        # return get_running_df(data, stimulus_timestamps)


    @classmethod
    def from_nwbfile(cls, nwbfile, **kwargs):
        obj = cls(path=None, **kwargs)
        obj._nwbfile = nwbfile
        return obj
