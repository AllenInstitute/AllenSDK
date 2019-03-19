import pynwb
from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule

def add_running_speed_to_nwbfile(nwbfile, running_speed, name='speed', unit='cm/s'):
    ''' Adds running speed data to an NWBFile as a timeseries in acquisition

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File to which runnign speeds will be written
    running_speed : RunningSpeed
        Contains attributes 'values' and 'timestamps'
    name : str, optional
        used as name of timeseries object
    unit : str, optional
        SI units of running speed values

    Returns
    -------
    nwbfile : pynwb.NWBFile

    '''

    timestamps_ts = TimeSeries(
        name='timestamps',
        timestamps=running_speed.timestamps,
        unit='s'
    )

    running_speed_series = pynwb.base.TimeSeries(
        name=name,
        data=running_speed.values,
        timestamps=timestamps_ts,
        unit=unit
    )

    running_mod = ProcessingModule('running', 'Running speed processing module')
    nwbfile.add_processing_module(running_mod)

    running_mod.add_data_interface(timestamps_ts)
    running_mod.add_data_interface(running_speed_series)

    nwbfile.add_acquisition(running_speed_series)
    return nwbfile


def add_running_data_df_to_nwbfile(nwbfile, running_data_df, unit_dict):
    ''' Adds running speed data to an NWBFile as timeseries in acquisition and processing

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File to which runnign speeds will be written
    running_speed : pandas.DataFrame
        Contains 'speed' and 'times', 'v_in', 'vsig', 'dx'
    unit : str, optional
        SI units of running speed values

    Returns
    -------
    nwbfile : pynwb.NWBFile

    '''
    assert running_data_df.index.name == 'timestamps'

    timestamps_ts = TimeSeries(
        name='timestamps',
        timestamps=running_data_df.index.values,
        unit=unit_dict['timestamps']
    )

    running_dx_series = TimeSeries(
        name='dx',
        data=running_data_df['dx'].values,
        timestamps=timestamps_ts,
        unit=unit_dict['dx']
    )

    running_speed_series = TimeSeries(
        name='speed',
        data=running_data_df['speed'].values,
        timestamps=timestamps_ts,
        unit=unit_dict['speed']
    )

    v_sig = TimeSeries(
        name='v_sig',
        data=running_data_df['v_sig'].values,
        timestamps=timestamps_ts, 
        unit=unit_dict['v_sig']
    )

    v_in = TimeSeries(
        name='v_in',
        data=running_data_df['v_in'].values,
        timestamps=timestamps_ts,
        unit=unit_dict['v_in']
    )

    running_mod = ProcessingModule('running', 'Running speed processing module')
    nwbfile.add_processing_module(running_mod)

    running_mod.add_data_interface(timestamps_ts)
    running_mod.add_data_interface(running_speed_series)
    running_mod.add_data_interface(running_dx_series)

    nwbfile.add_acquisition(v_sig)
    nwbfile.add_acquisition(v_in)

    return nwbfile