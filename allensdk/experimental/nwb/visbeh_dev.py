from pynwb import NWBFile, NWBHDF5IO, TimeSeries
import os
import pandas as pd
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import datetime
from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter

def add_epochs(nwbfile, ts_list, core_data):

    # REFACTOR to stimulus?

    stimulus_table = core_data['visual_stimuli']
    timestamps = core_data['time']

    for ri, row_series in stimulus_table.iterrows():
        row = row_series.to_dict()
        start_time = timestamps[int(row.pop('frame'))]
        stop_time = timestamps[int(row.pop('end_frame'))]
        assert start_time == row.pop('time')

        nwbfile.create_epoch(start_time=start_time,
                         stop_time=stop_time,
                         timeseries=ts_list,
                         tags='stimulus',
                         description='Stimulus Presentation Epoch',
                         metadata=row)




def test_visbeh_nwb(tmpdir_factory):

    data_dir = str(tmpdir_factory.mktemp("data"))
    save_file_name = os.path.join(data_dir, 'visbeh_test.nwb')

    foraging_file_name = '/allen/programs/braintv/production/neuralcoding/prod0/specimen_741992330/behavior_session_759661624/181001091719_411922_9aa2afc1-1142-42fa-98b3-838e7397fd40.pkl'
    
    visbeh_data = VisualBehaviorStimulusAdapter(foraging_file_name)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=visbeh_data.session_start_time,
        file_create_date=datetime.datetime.now()
    )

    nwbfile.add_acquisition(visbeh_data.running_speed)
    add_epochs(nwbfile, [visbeh_data.running_speed], visbeh_data.core_data)


    with NWBHDF5IO(save_file_name, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(save_file_name, mode='r').read()
    print 
    print 'OK'


class TmpDirFactoryMock(object):

    def mktemp(self, arg):
        return '/home/nicholasc/projects/allensdk/issues/244'

if __name__ == "__main__":

    tmpdir_factory = TmpDirFactoryMock()

    test_visbeh_nwb(tmpdir_factory)