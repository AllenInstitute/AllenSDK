from datetime import datetime
from pynwb import NWBFile
import numpy as np
from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries


def add_time_series(nwb_data,notebook):
    """

    Parameters
    ----------
    nwb_data: NwbReader Class
        methods to access nwb data
    notebook: ipfx.LabNotebookReader Class
        methods to access notebook data from nwb file
    Returns
    -------
    nwbfile: pynwb.NWBFile object with timeseries data

    """

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

        # Container for metadata that does not have an obvious place in pynwb timeseries
        # These properties actually are not used by the ipfx code
        sweep_metadata = {}

        scale_factor = notebook.get_value("Scale Factor", sweep_number, None)
        if scale_factor is None:
            raise Exception("Unable to read scale factor for " + sweep_name)
        sweep_metadata["stimulus_scale_factor"] = scale_factor
        cnt = notebook.get_value("Set Sweep Count", sweep_number, 0)
        stim_code_ext = acquisition["stimulus_description"] + "[%d]" % int(cnt)
        sweep_metadata["stimulus_code_ext"] = stim_code_ext

        # ------------------------------------------------------------------------------

        if stimulus["clamp_mode"] == "voltage_clamp":

            stimulus_series = VoltageClampStimulusSeries(
                name=sweep_name, source=stimulus['source'], data=stimulus['data'], unit=stimulus['unit'],
                conversion = stimulus['conversion'],
                starting_time=np.nan,
                electrode=elec, gain=np.nan,
                rate=stimulus['rate'],
            )
        elif stimulus["clamp_mode"] == "current_clamp":

            stimulus_series = CurrentClampStimulusSeries(
                name=sweep_name, source=stimulus['source'], data=stimulus['data'], unit=stimulus['unit'],
                conversion = stimulus['conversion'],
                starting_time=np.nan,
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
                unit=acquisition['unit'], conversion=acquisition['conversion'], resolution=np.nan, starting_time=np.nan,
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




