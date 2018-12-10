
from pynwb import NWBFile, NWBHDF5IO
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import datetime
import os
from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter


def main(save_file_name, foraging_file_name):
    
    visbeh_data = VisualBehaviorStimulusAdapter(foraging_file_name)

    nwbfile = NWBFile(
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=visbeh_data.session_start_time,
        file_create_date=datetime.datetime.now(),
        epochs = visbeh_data.stimulus_epoch_table
    )

    nwbfile.add_acquisition(visbeh_data.running_speed)
    
    for x in visbeh_data.image_series_list:
        nwbfile.add_stimulus_template(x)
    
    for y in visbeh_data.index_series_list:
        nwbfile.add_stimulus(y)

    with NWBHDF5IO(save_file_name, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(save_file_name, mode='r').read()


if __name__ == "__main__":

    data_dir = '//allen/aibs/technology/nicholasc'

    save_file_name = os.path.join(data_dir, 'visbeh_example_1.nwb')
    foraging_file_name = "/allen/programs/braintv/production/visualbehavior/prod0/specimen_738786518/behavior_session_759866491/181002090744_403468_7add2e7c-96fd-4aa0-b864-3dc4d4c38efa.pkl"
    main(save_file_name, foraging_file_name)

    foraging_file_name = '/allen/programs/braintv/production/neuralcoding/prod0/specimen_738720433/behavior_session_760658830/181004091143_409296_2e5b5a55-af4b-4f94-829d-0048df1eb550.pkl'
    save_file_name = os.path.join(data_dir, 'visbeh_example_2.nwb')
    main(save_file_name, foraging_file_name)