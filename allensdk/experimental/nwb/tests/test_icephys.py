import os
from allensdk.experimental.nwb.icephys import nwb1_reader, build_nwb2_file
import ipfx.lab_notebook_reader as lab_notebook_reader
import numpy as np
import pytest
from pynwb import NWBHDF5IO

NWB1_FILE_NAMES =[
                    "/allen/aibs/technology/sergeyg/ephys_pipeline/nwb_conversion/nwb1/Npr3-IRES2-CreSst-IRES-FlpOAi65-401243.04.01.01.nwb",
                    "/allen/aibs/technology/sergeyg/ephys_pipeline/nwb_conversion/nwb1/Pvalb-IRES-Cre;Ai14-406663.04.01.01.nwb"
                 ]


def make_nwb2_file_name(nwb1_file_name,test_dir):

    base_name = os.path.basename(nwb1_file_name)
    nwb2_file_name = os.path.join(test_dir, base_name)

    return nwb2_file_name

@pytest.mark.parametrize('nwb1_file_name', NWB1_FILE_NAMES)
def test_time_series(nwb1_file_name, tmpdir_factory):

    nwb1_data = nwb1_reader.create_nwb_reader(nwb1_file_name)
    notebook = lab_notebook_reader.create_lab_notebook_reader(nwb1_file_name)

    nwb2file = build_nwb2_file.add_time_series(nwb1_data, notebook)

    test_dir = str(tmpdir_factory.mktemp("test"))
    nwb2_file_name = make_nwb2_file_name(nwb1_file_name,test_dir)

    with NWBHDF5IO(nwb2_file_name, mode='w') as io:
        io.write(nwb2file)

    nwbfile = NWBHDF5IO(nwb2_file_name, mode='r').read() # load file back into memory

    sweep_names = nwbfile.acquisition.keys()

    for sweep_name in sweep_names:

        sweep_number = nwb1_data.get_sweep_number(sweep_name)

        acquisition1 = nwb1_data.get_acquisition(sweep_number)
        stimulus1 = nwb1_data.get_stimulus(sweep_number)

        acquisition2 = nwbfile.get_acquisition(sweep_name)
        stimulus2 = nwbfile.get_stimulus(sweep_name)

        assert(np.allclose(acquisition1['data'], acquisition2.data))
        assert(np.allclose(stimulus1['data'], stimulus2.data))


