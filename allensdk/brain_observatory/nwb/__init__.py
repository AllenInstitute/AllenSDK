import pynwb


def add_running_speed_to_nwbfile(nwbfile, running_speed, name='running_speed', unit='cm/s'):
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

    running_speed_series = pynwb.base.TimeSeries(
        name=name,
        data=running_speed.values,
        timestamps=running_speed.timestamps,
        unit=unit
    )
    nwbfile.add_acquisition(running_speed_series)
    return nwbfile
