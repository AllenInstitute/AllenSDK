from pynwb import NWBFile, NWBHDF5IO, TimeSeries
import os
import pandas as pd
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import datetime
from allensdk.experimental.nwb.stimulus import VisualBehaviorStimulusAdapter


def test_visbeh_nwb(tmpdir_factory):

    data_dir = str(tmpdir_factory.mktemp("data"))
    save_file_name = os.path.join(data_dir, 'visbeh_test.nwb')

    foraging_file_name = "/allen/programs/braintv/production/visualbehavior/prod0/specimen_738786518/behavior_session_759866491/181002090744_403468_7add2e7c-96fd-4aa0-b864-3dc4d4c38efa.pkl"
    
    visbeh_data = VisualBehaviorStimulusAdapter(foraging_file_name)

    epoch_table = visbeh_data.get_epoch_table()

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=visbeh_data.session_start_time,
        file_create_date=datetime.datetime.now(),
        epochs = epoch_table
    )

    nwbfile.add_acquisition(visbeh_data.running_speed)

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