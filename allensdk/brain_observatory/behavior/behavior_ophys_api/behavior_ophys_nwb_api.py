import datetime
from pynwb import NWBFile, NWBHDF5IO
import pandas as pd
import allensdk.brain_observatory.nwb as nwb
import numpy as np
import SimpleITK as sitk

from allensdk.brain_observatory.nwb.nwb_api import NwbApi
from allensdk.brain_observatory.behavior.trials_processing import TRIAL_COLUMN_DESCRIPTION_DICT


class BehaviorOphysNwbApi(NwbApi):

    def save(self, session_object):

        nwbfile = NWBFile(
            session_description=str(session_object.metadata['session_type']),
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata['experiment_date'],
            file_create_date=datetime.datetime.now()
        )

        # Add stimulus_timestamps to NWB in-memory object:
        nwb.add_stimulus_timestamps(nwbfile, session_object.stimulus_timestamps)

        # Add running data to NWB in-memory object:
        unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
        nwb.add_running_data_df_to_nwbfile(nwbfile, session_object.running_data_df, unit_dict)

        # Add ophys to NWB in-memory object:
        nwb.add_ophys_timestamps(nwbfile, session_object.ophys_timestamps)

        # Add stimulus template data to NWB in-memory object:
        for name, image_data in session_object.stimulus_templates.items():
            nwb.add_stimulus_template(nwbfile, image_data, name)

            # Add index for this template to NWB in-memory object:
            nwb_template = nwbfile.stimulus_template[name]
            stimulus_index = session_object.stimulus_index[session_object.stimulus_index['image_set'] == nwb_template.name]
            nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        # Add stimulus presentations data to NWB in-memory object:
        nwb.add_stimulus_presentations(nwbfile, session_object.stimulus_presentations)

        # Add trials data to NWB in-memory object:
        nwb.add_trials(nwbfile, session_object.trials, TRIAL_COLUMN_DESCRIPTION_DICT)

        # Add licks data to NWB in-memory object:
        nwb.add_licks(nwbfile, session_object.licks)

        # Add rewards data to NWB in-memory object:
        nwb.add_rewards(nwbfile, session_object.rewards)

        # Add max_projection image data to NWB in-memory object:
        nwb.add_max_projection(nwbfile, session_object.max_projection)

        # Add average_image image data to NWB in-memory object:
        nwb.add_average_image(nwbfile, session_object.average_image)

        # Write the file:
        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_running_data_df(self, **kwargs):

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

    def get_stimulus_templates(self, **kwargs):
        return {key: val.data[:] for key, val in self.nwbfile.stimulus_template.items()}

    def get_ophys_timestamps(self) -> np.ndarray:
        return self.nwbfile.modules['two_photon_imaging'].get_data_interface('timestamps').timestamps

    def get_stimulus_timestamps(self) -> np.ndarray:
        return self.nwbfile.modules['stimulus'].get_data_interface('timestamps').timestamps

    def get_trials(self) -> pd.DataFrame:
        trials = self.nwbfile.trials.to_dataframe()
        trials.index = trials.index.rename('trials_id')
        return trials

    def get_licks(self) -> np.ndarray:
        return pd.DataFrame({'time': self.nwbfile.modules['licking'].get_data_interface('licks')['timestamps'].timestamps[:]})

    def get_rewards(self) -> np.ndarray:
        time = self.nwbfile.modules['rewards'].get_data_interface('timestamps').timestamps[:]
        autorewarded = self.nwbfile.modules['rewards'].get_data_interface('autorewarded').data[:]
        volume = self.nwbfile.modules['rewards'].get_data_interface('volume').data[:]

        return pd.DataFrame({'time': time, 'volume': volume, 'autorewarded': autorewarded})

    def get_max_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('max_projection', 'two_photon_imaging', image_api=image_api)

    def get_average_image(self, image_api=None) -> sitk.Image:
        return self.get_image('average_image', 'two_photon_imaging', image_api=image_api)

    def get_stimulus_index(self) -> pd.DataFrame:

        data_dict = {'timestamps': [], 'image_set': [], 'image_index': []}
        for stimulus_name in self.nwbfile.stimulus:
            curr_image_index_series = self.nwbfile.stimulus[stimulus_name]
            data_dict['image_set'] += [stimulus_name] * len(curr_image_index_series.data[:])
            data_dict['image_index'] += list(curr_image_index_series.data[:])
            data_dict['timestamps'] += list(curr_image_index_series.timestamps[:])

        stimulus_index_df = pd.DataFrame(data_dict)
        stimulus_index_df.set_index('timestamps', inplace=True)
        stimulus_index_df.sort_index(inplace=True)
        return stimulus_index_df
