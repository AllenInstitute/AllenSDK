import datetime
from pynwb import NWBFile, NWBHDF5IO
import pandas as pd
import allensdk.brain_observatory.nwb as nwb

from allensdk.brain_observatory.nwb.nwb_api import NwbApi


class BehaviorOphysNwbApi(NwbApi):

    def save(self, session_object):

        nwbfile = NWBFile(
            session_description=str(session_object.metadata['session_type']),
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata['experiment_date'],
            file_create_date=datetime.datetime.now()
        )

        unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        nwb.add_running_data_df_to_nwbfile(nwbfile, session_object.running_data_df, unit_dict)

        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_running_data_df(self, **kwargs):

        print(self.nwbfile.modules)

        running_speed = self.get_running_speed()

        running_data_df = pd.DataFrame({'speed': running_speed.values},
                                       index=pd.Index(running_speed.timestamps, name='timestamps'))

        for key in ['v_in', 'v_sig']:
            if key in self.nwbfile.acquisition:
                running_data_df[key] = self.nwbfile.get_acquisition(key).data

        for key in ['dx']:
            if ('running' in self.nwbfile.modules) and (key in self.nwbfile.modules['running'].fields['data_interfaces']):
                running_data_df[key] = self.nwbfile.modules['running'].get_data_interface(key).data

        return running_data_df

    def get_metadata(self, **kwargs):
        pass
