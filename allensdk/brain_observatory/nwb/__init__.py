from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralTimeSeries

def add_running_speed_to_nwbfile(nwbfile, running_df, unit='cm/s'):
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

    # print running_df.columns

    timestamps_ts = TimeSeries(
        name='running_speed_timestamps',
        timestamps=running_df['time'].values, 
        unit=unit
    )

    running_dx_series = TimeSeries(
        name='running_dx',
        data=running_df['dx'].values, 
        timestamps=timestamps_ts, 
        unit=unit
    )

    running_speed_series = TimeSeries(
        name='running_speed',
        data=running_df['speed'].values, 
        timestamps=timestamps_ts, 
        unit=unit
    )

    v_sig = TimeSeries(
        name='v_sig',
        data=running_df['v_sig'].values, 
        timestamps=timestamps_ts, 
        unit=unit
    )

    v_in = TimeSeries(
        name='v_in',
        data=running_df['v_in'].values, 
        timestamps=timestamps_ts, 
        unit=unit
    )

    running_bts = BehavioralTimeSeries()
    nwbfile.add_analysis(running_bts)

    running_bts.add_timeseries(timestamps_ts)
    running_bts.add_timeseries(running_speed_series)
    running_bts.add_timeseries(running_dx_series)

    nwbfile.add_acquisition(v_sig)
    nwbfile.add_acquisition(v_in)

    return nwbfile