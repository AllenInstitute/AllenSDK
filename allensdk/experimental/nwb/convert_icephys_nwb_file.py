from datetime import datetime
from pynwb import NWBFile
import ipfx.nwb_reader as nwb_reader
from pynwb import NWBHDF5IO
import icephys
import logging

NWB1_FILE_NAME = "/allen/programs/celltypes/production/mousecelltypes/prod589/Ephys_Roi_Result_500844779/500844779.nwb"
NWB2_FILE_NAME = "/local1/ephys/ivscc/nwb2/500844779_ver2.nwb"


def save_nwb2_file(nwb2file,nwb2_file_name):

    io = NWBHDF5IO(nwb2_file_name, 'w')
    io.write(nwb2file)
    io.close()


def load_nwb2_file(nwb2_file_name):

    io = NWBHDF5IO(nwb2_file_name, 'r')

    return io.read()


def main():

    logging.basicConfig(level="INFO")
    nwb2_file_name = NWB2_FILE_NAME
    nwb1_file_name = NWB1_FILE_NAME

    nwb_data = nwb_reader.create_nwb_reader(nwb1_file_name)

    nwb2file = NWBFile('pynwb sprint', 'example file for kitware', 'EXAMPLE_ID', datetime.now(),
                       lab='Intracellular Ephys Lab',
                       institution='Allen Institute',
                       experiment_description='IVSCC recording',
                       file_create_date=datetime.now()
    )

    icephys.add_time_series(nwb2file,nwb_data)
    logging.info("Added time series")

    save_nwb2_file(nwb2file,nwb2_file_name)
    logging.info("Saved the nwb file")

    load_nwb2_file(nwb2_file_name)
    logging.info("Loaded back the nwb file")

if __name__ == "__main__": main()
