import datetime
from pynwb import NWBFile, NWBHDF5IO
import pandas as pd
from allensdk.brain_observatory.nwb import add_running_data_df_to_nwbfile
from allensdk.brain_observatory import RunningSpeed


class BehaviorOphysNwbApi(object):

    def __init__(self, nwb_filepath):

        self.nwb_filepath = nwb_filepath

    def save(self, session_object):

        nwbfile = NWBFile(
            session_description=str(session_object.metadata['session_type']),
            identifier=str(session_object.metadata['behavior_session_uuid']),
            session_start_time=session_object.metadata['experiment_date'],
            file_create_date=datetime.datetime.now()
        )

        unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        add_running_data_df_to_nwbfile(nwbfile, session_object.running_data_df, unit_dict)

        with NWBHDF5IO(self.nwb_filepath, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

    def get_running_data_df(self, ophys_experiment_id=None, **kwargs):
        with NWBHDF5IO(self.nwb_filepath, 'r') as nwb_file_reader:
            obtained = nwb_file_reader.read()

            print (obtained.acquisition)
            print (obtained.analysis)
            raise

            return pd.DataFrame({'v_sig': obtained.get_acquisition('v_sig').data.value,
                                 'v_in': obtained.get_acquisition('v_in').data.value})

    def get_metadata(self, ophys_experiment_id=None, **kwargs):
        pass

    def get_running_speed(self, ophys_experiment_id=None, **kwargs):
        
        running_data_df = self.get_running_data_df(ophys_experiment_id=ophys_experiment_id, **kwargs)
        return RunningSpeed(timestamps=running_data_df.index.values,
                            values=running_data_df.speed.values)


        # stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        # behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        # data = pd.read_pickle(behavior_stimulus_file)
        # return get_running_df(data, stimulus_timestamps)

    # def get_running_speed(self, ophys_experiment_id=None):
    #     running_data_df = self.get_running_data_df(ophys_experiment_id=ophys_experiment_id)
    #     return running_data_df.time.values, running_data_df.speed.values