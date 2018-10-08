import sys
import nwb1_reader
import logging
import ipfx.lab_notebook_reader as lab_notebook_reader
import ic_ephys
import os

DEFAULT_NWB1_FILE_NAME ="/allen/aibs/technology/sergeyg/ephys_pipeline/nwb_conversion/nwb1/Npr3-IRES2-CreSst-IRES-FlpOAi65-401243.04.01.01.nwb"


def main():

    """
    Usage:
    $python icephys_nwb1_to_nwb2.py NWB1_FILE_NAME
    """
    logging.basicConfig(level="INFO")

    if len(sys.argv) == 1:
        sys.argv.append(DEFAULT_NWB1_FILE_NAME)

    nwb1_file_name = sys.argv[1]
    dir_name = os.path.dirname(nwb1_file_name)
    base_name = os.path.basename(nwb1_file_name)
    file_name, file_extension = os.path.splitext(base_name)
    nwb2_file_name = os.path.join(dir_name, file_name+"_ver2" + file_extension)


    nwb_data = nwb1_reader.create_nwb_reader(nwb1_file_name)
    notebook = lab_notebook_reader.create_lab_notebook_reader(nwb1_file_name)

    nwb2file = ic_ephys.build_nwb2file(nwb_data,notebook)
    logging.info("Created nwb2 file")

    ic_ephys.save_nwb2_file(nwb2file,nwb2_file_name)
    logging.info("Saved the nwb2 file")

    ic_ephys.load_nwb2_file(nwb2_file_name)
    logging.info("Loaded back the nwb2 file")

if __name__ == "__main__": main()
