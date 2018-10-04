from datetime import datetime
from pynwb import NWBFile
import nwb1_reader
from pynwb import NWBHDF5IO
import icephys
import logging
import numpy as np
from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
import time
import ipfx.lab_notebook_reader as lab_notebook_reader

#NWB1_FILE_NAME = "/allen/programs/celltypes/production/mousecelltypes/prod589/Ephys_Roi_Result_500844779/500844779.nwb"
#NWB2_FILE_NAME = "/local1/ephys/ivscc/nwb2/500844779_ver2.nwb"
H5_FILE_NAME = None
NWB1_FILE_NAME ="/local1/ephys/patchseq/nwb2/Npr3-IRES2-CreSst-IRES-FlpOAi65-401243.04.01.01.nwb"
NWB2_FILE_NAME = "/local1/ephys/patchseq/nwb2/patchseq_nwb_ver2.nwb"


def build_nwb2file(nwb_data,notebook):


    nwb2file = NWBFile('pynwb sprint', 'example file for kitware', 'EXAMPLE_ID', datetime.now(),
                       lab='Intracellular Ephys Lab',
                       institution='Allen Institute',
                       experiment_description='IVSCC recording',
                       file_create_date=datetime.now()
    )


    device = nwb2file.create_device(name='electrode_0', source='a source')

    elec = nwb2file.create_ic_electrode(
        name="elec0", source='PyNWB tutorial example',
        description=' some kind of electrode',
        device=device)

    for sweep_name in nwb_data.get_sweep_names():

        sweep_number = nwb_data.get_sweep_number(sweep_name)
        acquisition = nwb_data.get_acquisition(sweep_number)
        stimulus = nwb_data.get_stimulus(sweep_number)

        if stimulus["clamp_mode"] == "voltage_clamp":

            stimulus_series = VoltageClampStimulusSeries(
                name=sweep_name, source=stimulus['source'], data=stimulus['data'], unit=stimulus['unit'],
                starting_time=np.nan, rate=stimulus['rate'], electrode=elec,
                gain=np.nan,
            )
        elif stimulus["clamp_mode"] == "current_clamp":

            stimulus_series = CurrentClampStimulusSeries(
                name=sweep_name, source=stimulus['source'], data=stimulus['data'], unit=stimulus['unit'],
                electrode=elec, gain=np.nan,
                rate=stimulus['rate'],
            )

        if acquisition["clamp_mode"] == "voltage_clamp":

            acquisition_series = VoltageClampSeries(
                name=sweep_name, source=acquisition['source'], data=acquisition['data'],
                unit=acquisition['unit'], conversion=acquisition['conversion'],
                resolution=np.nan, starting_time=np.nan, rate=acquisition['rate'],
                electrode=elec, gain=np.nan, capacitance_slow=np.nan, resistance_comp_correction=np.nan,
                capacitance_fast=np.nan, resistance_comp_bandwidth=np.nan, resistance_comp_prediction=np.nan,
                whole_cell_capacitance_comp=np.nan, whole_cell_series_resistance_comp=np.nan,
                stimulus_description=acquisition["stimulus_description"]
            )
        elif acquisition["clamp_mode"] == "current_clamp":

            bridge_balance = notebook.get_value("Bridge Bal Value", sweep_number, np.nan)
            bias_current = notebook.get_value("I-Clamp Holding Level", sweep_number, np.nan)
            acquisition_series = CurrentClampSeries(
                name=sweep_name, source=acquisition['source'], data=acquisition['data'],
                unit=acquisition['unit'], conversion=np.nan, resolution=np.nan, starting_time=np.nan,
                rate=acquisition['rate'],
                electrode=elec, gain=np.nan,
                bias_current=bias_current,
                bridge_balance=bridge_balance,
                stimulus_description=acquisition["stimulus_description"],
                capacitance_compensation=np.nan
            )

        nwb2file.add_stimulus(stimulus_series)
        nwb2file.add_acquisition(acquisition_series)

    return nwb2file


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
    h5_file_name = H5_FILE_NAME

    nwb_data = nwb1_reader.create_nwb_reader(nwb1_file_name)
    notebook = lab_notebook_reader.create_lab_notebook_reader(nwb1_file_name, h5_file_name)

    nwb2file = build_nwb2file(nwb_data,notebook)
    logging.info("Created nwb2 file")

    save_nwb2_file(nwb2file,nwb2_file_name)
    logging.info("Saved the nwb2 file")

    load_nwb2_file(nwb2_file_name)
    logging.info("Loaded back the nwb2 file")

if __name__ == "__main__": main()
