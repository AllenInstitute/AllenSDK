import numpy as np
from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries



def extract_sweep_meta_data(nwb_data,sweep_name):

    sweep_info = {}
    attrs = nwb_data.get_sweep_attrs(sweep_name)
    ancestry = attrs["ancestry"]
    sweep_info['clamp_mode'] = ancestry[-1]
    sweep_info['sweep_number'] = nwb_data.get_sweep_number(sweep_name)
    sweep_info['stimulus_code'] = nwb_data.get_stim_code(sweep_name)
    sweep_info['sweep_name'] = sweep_name

    if "CurrentClamp".encode() in ancestry[-1]:
        sweep_info['stimulus_units'] = 'pA'
        sweep_info['clamp_mode'] = 'CurrentClamp'
    elif "VoltageClamp".encode() in ancestry[-1]:
        sweep_info['stimulus_units'] = 'mV'
        sweep_info['clamp_mode'] = 'VoltageClamp'
    else:
        raise Exception("Unable to determine clamp mode in " + sweep_name)

    return sweep_info


def add_time_series(nwb2file,nwb_data):

    device = nwb2file.create_device(name='Heka ITC-1600', source='a source')

    elec = nwb2file.create_ic_electrode(
        name="elec0", source='PyNWB tutorial example',
        description='a mock intracellular electrode',
        device=device)

    for sweep_name in nwb_data.get_sweep_names():

        sweep_info = extract_sweep_meta_data(nwb_data,sweep_name)
        sweep_number = nwb_data.get_sweep_number(sweep_name)
        sweep_data = nwb_data.get_sweep_data(sweep_number)

        response = sweep_data['response']
        stimulus = sweep_data['stimulus']

        if sweep_info['clamp_mode'] == "VoltageClamp":  # voltage clamp

            stimulus_series = VoltageClampStimulusSeries(
                name=sweep_name, source="command", data=stimulus, unit='V',
                starting_time=123.6, rate=10e3, electrode=elec, gain=0.02)

            response_series = VoltageClampSeries(
                name=sweep_name, source="command", data=response,
                unit='A', conversion=1e-12, resolution=np.nan, starting_time=123.6, rate=20e3,
                electrode=elec, gain=0.02, capacitance_slow=100e-12, resistance_comp_correction=70.0,
                capacitance_fast=np.nan, resistance_comp_bandwidth=np.nan, resistance_comp_prediction=np.nan,
                whole_cell_capacitance_comp=np.nan, whole_cell_series_resistance_comp=np.nan)

        elif sweep_info['clamp_mode'] == "CurrentClamp":

            stimulus_series = CurrentClampStimulusSeries(
                name=sweep_name, source="command", data=stimulus, unit='A',
                starting_time=123.6, rate=10e3, electrode=elec, gain=0.02)

            response_series = CurrentClampSeries(
                name=sweep_name, source="command", data=response,
                unit='V', conversion=1e-12, resolution=np.nan, starting_time=123.6, rate=20e3,
                electrode=elec, gain=0.02,
                bias_current=np.nan, bridge_balance=np.nan,


                capacitance_compensation=np.nan)
        else:
            raise Exception("Unknown clamp mode")

        nwb2file.add_stimulus(stimulus_series)
        nwb2file.add_acquisition(response_series)






