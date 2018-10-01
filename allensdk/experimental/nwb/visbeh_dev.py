from pynwb import NWBFile, NWBHDF5IO, TimeSeries
import os
import pandas as pd
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import datetime

def get_running_speed(core_data):

    tmp_running_df = core_data['running']
    running_data = tmp_running_df['speed'] 
    timestamps = tmp_running_df['time'] 

    running_speed = TimeSeries(
        name='running_speed',
        source='Allen Brain Observatory: Visual Coding',
        data=running_data.values,
        timestamps=timestamps.values,
        unit='cm/s')

    return running_speed

def add_epochs(nwbfile, ts_list, core_data):

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
    data = pd.read_pickle(foraging_file_name)
    core_data = data_to_change_detection_core(data)

    running_speed_ts = get_running_speed(core_data)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=core_data['metadata']['startdatetime'],
        file_create_date=datetime.datetime.now()
    )


    nwbfile.add_acquisition(running_speed_ts)

    add_epochs(nwbfile, [running_speed_ts], core_data)


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