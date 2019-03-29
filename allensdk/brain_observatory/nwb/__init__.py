import numpy as np
import datetime
import uuid
import pynwb
from pynwb.base import TimeSeries, Images
from pynwb.behavior import BehavioralEvents
from pynwb import ProcessingModule
from pynwb.image import ImageSeries, GrayscaleImage, IndexSeries

from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.image_api import ImageApi
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension


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

    return nwbfile


def add_running_data_df_to_nwbfile(nwbfile, running_data_df, unit_dict, index_key='timestamps'):
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
    assert running_data_df.index.name == index_key

    running_speed = RunningSpeed(timestamps=running_data_df.index.values,
                                 values=running_data_df['speed'].values)

    add_running_speed_to_nwbfile(nwbfile, running_speed, name='speed', unit=unit_dict['speed'])

    running_mod = nwbfile.modules['running']
    timestamps_ts = running_mod.get_data_interface('timestamps')

    running_dx_series = TimeSeries(
        name='dx',
        data=running_data_df['dx'].values,
        timestamps=timestamps_ts,
        unit=unit_dict['dx']
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

    running_mod.add_data_interface(running_dx_series)
    nwbfile.add_acquisition(v_sig)
    nwbfile.add_acquisition(v_in)

    return nwbfile


def add_stimulus_template(nwbfile, image_data, name):

    image_index = list(range(image_data.shape[0]))
    visual_stimulus_image_series = ImageSeries(name=name,
                                               data=image_data,
                                               unit='NA',
                                               format='raw',
                                               timestamps=image_index)

    nwbfile.add_stimulus_template(visual_stimulus_image_series)
    return nwbfile


def add_stimulus_presentations(nwbfile, stimulus_table, tag='stimulus_epoch'):
    ''' Adds a stimulus table (defining stimulus characteristics for each time point in a session) to an nwbfile as epochs.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
    stimulus_table: pd.DataFrame
        Each row corresponds to an epoch of time. Columns define the epoch (start and stop time) and its characteristics. 
        Nans will be replaced with the empty string. Required columns are:
            start_time :: the time at which this epoch started
            stop_time :: the time  at which this epoch ended
    tag : str, optional
        Each epoch in an nwb file has one or more tags. This string will be applied as a tag to all epochs created here

    Returns
    -------
    nwbfile : pynwb.NWBFile

    '''
    stimulus_table = stimulus_table.copy()

    ts = nwbfile.modules['stimulus'].get_data_interface('timestamps')

    for colname, series in stimulus_table.items():
        types = set(series.map(type))
        if len(types) > 1 and str in types:
            series.fillna('', inplace=True)
            stimulus_table[colname] = series.transform(str)

    stimulus_table['tags'] = [(tag,)] * stimulus_table.shape[0]
    stimulus_table['timeseries'] = [(ts,)] * stimulus_table.shape[0]

    container = pynwb.epoch.TimeIntervals.from_dataframe(stimulus_table, 'epochs')
    nwbfile.epochs = container

    return nwbfile


def add_ophys_timestamps(nwbfile, ophys_timestamps, module_name='two_photon_imaging'):

    stimulus_ts = TimeSeries(
        name='timestamps',
        timestamps=ophys_timestamps,
        unit='s'
    )

    stim_mod = ProcessingModule(module_name, 'Ophys timestamps processing module')
    nwbfile.add_processing_module(stim_mod)
    stim_mod.add_data_interface(stimulus_ts)

    return nwbfile


def add_stimulus_timestamps(nwbfile, stimulus_timestamps, module_name='stimulus'):

    stimulus_ts = TimeSeries(
        name='timestamps',
        timestamps=stimulus_timestamps,
        unit='s'
    )

    stim_mod = ProcessingModule(module_name, 'Stimulus Times processing')

    nwbfile.add_processing_module(stim_mod)
    stim_mod.add_data_interface(stimulus_ts)

    return nwbfile


def add_trials(nwbfile, trials, description_dict={}):

    order = list(trials.index)
    for _, row in trials[['start_time', 'stop_time']].iterrows():
        row_dict = row.to_dict()
        nwbfile.add_trial(**row_dict)

    for c in [c for c in trials.columns if c not in ['start_time', 'stop_time']]:
        index, data = dict_to_indexed_array(trials[c].to_dict(), order)
        if data.dtype == '<U1':
            data = trials[c].values
        if not len(data) == len(order):
            nwbfile.add_trial_column(name=c, description=description_dict.get(c, 'NOT IMPLEMENTED: %s' % c), data=data, index=index)
        else:
            nwbfile.add_trial_column(name=c, description=description_dict.get(c, 'NOT IMPLEMENTED: %s' % c), data=data)


def add_licks(nwbfile, licks):

    licks_event_series = TimeSeries(
        name='timestamps',
        timestamps=licks.time.values,
        unit='s'
    )

    # Add lick event timeseries to lick interface:
    licks_interface = BehavioralEvents([licks_event_series], 'licks')

    # Add lick interface to nwb file, by way of a processing module:
    licks_mod = ProcessingModule('licking', 'Licking behavior processing module')
    licks_mod.add_data_interface(licks_interface)
    nwbfile.add_processing_module(licks_mod)

    return nwbfile


def add_rewards(nwbfile, rewards_df):

    reward_timestamps_ts = TimeSeries(
        name='timestamps',
        timestamps=rewards_df.time.values,
        unit='s'
    )

    reward_volume_ts = TimeSeries(
        name='volume',
        data=rewards_df.volume.values,
        timestamps=reward_timestamps_ts,
        unit='ml'
    )

    autorewarded_ts = TimeSeries(
        name='autorewarded',
        data=rewards_df.autorewarded.values,
        timestamps=reward_timestamps_ts,
        unit=None
    )

    rewards_mod = ProcessingModule('rewards', 'Licking behavior processing module')
    rewards_mod.add_data_interface(reward_timestamps_ts)
    rewards_mod.add_data_interface(reward_volume_ts)
    rewards_mod.add_data_interface(autorewarded_ts)
    nwbfile.add_processing_module(rewards_mod)

    return nwbfile


def add_image(nwbfile, image_data, image_name, module_name, module_description, image_api=None):

    description = '{} image at pixels/cm resolution'.format(image_name)

    if image_api is None:
        image_api = ImageApi

    data, spacing, unit = ImageApi.deserialize(image_data)
    assert spacing[0] == spacing[1] and len(spacing) == 2 and unit == 'mm'

    if module_name not in nwbfile.modules:
        ophys_mod = ProcessingModule(module_name, module_description)
        nwbfile.add_processing_module(ophys_mod)
    else:
        ophys_mod = nwbfile.modules[module_name]

    image = GrayscaleImage(image_name, data, resolution=spacing[0] / 10, description=description)

    if 'images' not in ophys_mod.containers:
        images = Images(name='images')
        ophys_mod.add_data_interface(images)
    else:
        images = ophys_mod['images']
    images.add_image(image)

    return nwbfile


def add_max_projection(nwbfile, max_projection, image_api=None):

    add_image(nwbfile, max_projection, 'max_projection', 'two_photon_imaging', 'Ophys timestamps processing module', image_api=image_api)


def add_average_image(nwbfile, average_image, image_api=None):

    add_image(nwbfile, average_image, 'average_image', 'two_photon_imaging', 'Ophys timestamps processing module', image_api=image_api)


def add_stimulus_index(nwbfile, stimulus_index, nwb_template):

    assert stimulus_index.index.name == 'timestamps'

    image_index = IndexSeries(
        name=nwb_template.name,
        data=stimulus_index['image_index'].values,
        unit='None',
        indexed_timeseries=nwb_template,
        timestamps=stimulus_index.index.values)
    nwbfile.add_stimulus(image_index)


def add_metadata(nwbfile, metadata):

    OphysBehaviorMetaData = load_LabMetaData_extension(OphysBehaviorMetaDataSchema, 'AIBS_ophys_behavior')
    metadata_clean = OphysBehaviorMetaDataSchema().dump(metadata)

    new_metadata_dict = {}
    for key, val in metadata_clean.items():
        if isinstance(val, list):
            new_metadata_dict[key] = np.array(val)
        elif isinstance(val, (datetime.datetime, uuid.UUID)):
            new_metadata_dict[key] = str(val)
        else:
            new_metadata_dict[key] = val
    nwb_metadata = OphysBehaviorMetaData(name='metadata', **new_metadata_dict)
    nwbfile.add_lab_meta_data(nwb_metadata)


def add_task_parameters(nwbfile, task_parameters):

    OphysBehaviorTaskParameters = load_LabMetaData_extension(OphysBehaviorTaskParametersSchema, 'AIBS_ophys_behavior')
    task_parameters_clean = OphysBehaviorTaskParametersSchema().dump(task_parameters)

    new_task_parameters_dict = {}
    for key, val in task_parameters_clean.items():
        if isinstance(val, list):
            new_task_parameters_dict[key] = np.array(val)
        else:
            new_task_parameters_dict[key] = val
    nwb_task_parameters = OphysBehaviorTaskParameters(name='task_parameters', **new_task_parameters_dict)
    nwbfile.add_lab_meta_data(nwb_task_parameters)
