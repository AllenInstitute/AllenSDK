from pynwb import NWBFile, NWBHDF5IO, TimeSeries
import os
import pandas as pd
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import datetime

def test_visbeh_nwb(tmpdir_factory):

    data_dir = str(tmpdir_factory.mktemp("data"))
    save_file_name = os.path.join(data_dir, 'visbeh_test.nwb')

    foraging_file_name = '/allen/programs/braintv/production/neuralcoding/prod0/specimen_741992330/behavior_session_759661624/181001091719_411922_9aa2afc1-1142-42fa-98b3-838e7397fd40.pkl'

    data = pd.read_pickle(foraging_file_name)
    core_data = data_to_change_detection_core(data)

    tmp_running_df = core_data['running']
    running_data = tmp_running_df['speed'] 
    timestamps = tmp_running_df['time'] 

    running_speed = TimeSeries(
        name='running_speed',
        source='Allen Brain Observatory: Visual Coding',
        data=running_data.values,
        timestamps=timestamps.values,
        unit='cm/s')



    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=core_data['metadata']['startdatetime'],
        file_create_date=datetime.datetime.now()
    )

    nwbfile.add_acquisition(running_speed)

    epoch_table = pd.DataFrame({'start':[0.,1.], 'end':[1.,2.], 'stimulus':['a', 'b']})

    for ri, row in epoch_table.iterrows():
        nwbfile.create_epoch(start_time=row.start,
                         stop_time=row.end,
                         timeseries=[running_speed],
                         tags='stimulus',
                         description=row.stimulus)


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