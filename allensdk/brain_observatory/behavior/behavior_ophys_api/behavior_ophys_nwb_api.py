import datetime
from pynwb import NWBFile, NWBHDF5IO
from allensdk.brain_observatory.nwb import add_running_speed_to_nwbfile

class BehaviorOphysNwbApi(object):

    def __init__(self, nwb_filepath):

        self.nwb_filepath = nwb_filepath

    def save(self, session_object):

        nwbfile = NWBFile(
            session_description='test foraging2',
            identifier='behavior_session_uuid',
            session_start_time=datetime.datetime.now(),
            file_create_date=datetime.datetime.now()
        )

        add_running_speed_to_nwbfile(nwbfile, session_object.running_speed)


        with NWBHDF5IO(self.nwb_filepath, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)
