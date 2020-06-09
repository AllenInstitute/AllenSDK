import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import datetime
import uuid
import SimpleITK as sitk
import pynwb
from pynwb.base import TimeSeries, Images
from pynwb.behavior import BehavioralEvents
from pynwb import ProcessingModule
from pynwb.image import ImageSeries, GrayscaleImage, IndexSeries
from pynwb.ophys import DfOverF, ImageSegmentation, OpticalChannel, Fluorescence

import allensdk.brain_observatory.roi_masks as roi
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.behavior.image_api import Image
from allensdk.brain_observatory.behavior.image_api import ImageApi
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension


log = logging.getLogger("allensdk.brain_observatory.nwb")


def read_eye_dlc_tracking_ellipses(input_path: Path) -> dict:
    """Reads eye tracking ellipse fit data from an h5 file.

    Args:
        input_path (Path): Path to eye tracking ellipse fit h5 file

    Returns:
        dict: Loaded h5 data. Each 'params' field contains dataframes with]
            ellipse fit parameters. Dataframes contain 5 columns each
            consisting of: "center_x", "center_y", "height", "phi", "width"
    """

    eye_dlc_tracking_data = {}

    # TODO: Some ellipses.h5 files have the 'cr' key as complex type instead of
    # float. For now, when loading ellipses.h5 files, always coerce to float
    # but this should eventually be resolved upstream...
    # See: allensdk.brain_observatory.eye_tracking
    pupil_params = pd.read_hdf(input_path, key="pupil").astype(float)
    cr_params = pd.read_hdf(input_path, key="cr").astype(float)
    eye_params = pd.read_hdf(input_path, key="eye").astype(float)

    eye_dlc_tracking_data["pupil_params"] = pupil_params
    eye_dlc_tracking_data["cr_params"] = cr_params
    eye_dlc_tracking_data["eye_params"] = eye_params

    return eye_dlc_tracking_data


def read_eye_gaze_mappings(input_path: Path) -> dict:
    """Reads eye gaze mapping data from an h5 file.

    Args:
        input_path (Path): Path to eye gaze mapping h5 data file produced by
            'allensdk.brain_observatory.gaze_mapping' module.

    Returns:
        dict: Loaded h5 data.
            *_eye_areas: Area of eye (in pixels^2) over time
            *_pupil_areas: Area of pupil (in pixels^2) over time
            *_screen_coordinates: y, x screen coordinates (in cm) over time
            *_screen_coordinates_spherical: y, x screen coordinates (in deg) over time
            synced_frame_timestamps: synced timestamps for video frames (in sec)
    """

    eye_gaze_data = {}

    eye_gaze_data["raw_eye_areas"] = pd.read_hdf(input_path, key="raw_eye_areas")
    eye_gaze_data["raw_pupil_areas"] = pd.read_hdf(input_path, key="raw_pupil_areas")
    eye_gaze_data["raw_screen_coordinates"] = pd.read_hdf(input_path, key="raw_screen_coordinates")
    eye_gaze_data["raw_screen_coordinates_spherical"] = pd.read_hdf(input_path, key="raw_screen_coordinates_spherical")

    eye_gaze_data["new_eye_areas"] = pd.read_hdf(input_path, key="new_eye_areas")
    eye_gaze_data["new_pupil_areas"] = pd.read_hdf(input_path, key="new_pupil_areas")
    eye_gaze_data["new_screen_coordinates"] = pd.read_hdf(input_path, key="new_screen_coordinates")
    eye_gaze_data["new_screen_coordinates_spherical"] = pd.read_hdf(input_path, key="new_screen_coordinates_spherical")

    eye_gaze_data["synced_frame_timestamps"] = pd.read_hdf(input_path, key="synced_frame_timestamps")

    return eye_gaze_data


def create_eye_gaze_mapping_dataframe(eye_gaze_data: dict) -> pd.DataFrame:

    eye_gaze_mapping_df = pd.DataFrame(
        {
            "raw_eye_area": eye_gaze_data["raw_eye_areas"].values,
            "raw_pupil_area": eye_gaze_data["raw_pupil_areas"].values,
            "raw_screen_coordinates_x_cm": eye_gaze_data["raw_screen_coordinates"]["x_pos_cm"].values,
            "raw_screen_coordinates_y_cm": eye_gaze_data["raw_screen_coordinates"]["y_pos_cm"].values,
            "raw_screen_coordinates_spherical_x_deg": eye_gaze_data["raw_screen_coordinates_spherical"]["x_pos_deg"].values,
            "raw_screen_coordinates_spherical_y_deg": eye_gaze_data["raw_screen_coordinates_spherical"]["y_pos_deg"].values,

            "filtered_eye_area": eye_gaze_data["new_eye_areas"].values,
            "filtered_pupil_area": eye_gaze_data["new_pupil_areas"].values,
            "filtered_screen_coordinates_x_cm": eye_gaze_data["new_screen_coordinates"]["x_pos_cm"].values,
            "filtered_screen_coordinates_y_cm": eye_gaze_data["new_screen_coordinates"]["y_pos_cm"].values,
            "filtered_screen_coordinates_spherical_x_deg": eye_gaze_data["new_screen_coordinates_spherical"]["x_pos_deg"].values,
            "filtered_screen_coordinates_spherical_y_deg": eye_gaze_data["new_screen_coordinates_spherical"]["y_pos_deg"].values
        },
        index=eye_gaze_data["synced_frame_timestamps"].values
    )
    return eye_gaze_mapping_df


def eye_tracking_data_is_valid(eye_dlc_tracking_data: dict,
                               synced_timestamps: pd.Series) -> bool:
    is_valid = True

    pupil_params = eye_dlc_tracking_data["pupil_params"]
    cr_params = eye_dlc_tracking_data["cr_params"]
    eye_params = eye_dlc_tracking_data["eye_params"]

    num_frames_match = ((pupil_params.shape[0] == cr_params.shape[0])
                        and (cr_params.shape[0] == eye_params.shape[0]))
    if not num_frames_match:
        log.warn("The number of frames for ellipse fits don't "
                 "match when they should. No ellipse fits will be written! "
                 f"pupil_params ({pupil_params.shape[0]}), "
                 f"cr_params ({cr_params.shape[0]}), "
                 f"eye_params ({eye_params.shape[0]})")
        is_valid = False

    if (pupil_params.shape[0] != len(synced_timestamps)):
        log.warn("The number of camera sync pulses in the "
                 f"sync file ({len(synced_timestamps)}) do not match "
                 "with the number of eye tracking frames "
                 f"({pupil_params.shape[0]})! No ellipse fits will be written!")
        is_valid = False

    return is_valid


def create_eye_tracking_nwb_processing_module(eye_dlc_tracking_data: dict,
                                              synced_timestamps: pd.Series) -> pynwb.ProcessingModule:

    # Top level container for eye tracking processed data
    eye_tracking_mod = pynwb.ProcessingModule(name='eye_tracking',
                                              description='Eye tracking processing module')

    # Data interfaces of dlc_fits_container
    pupil_fits = eye_dlc_tracking_data["pupil_params"].assign(timestamps=synced_timestamps)
    pupil_params = pynwb.core.DynamicTable.from_dataframe(df=pupil_fits,
                                                          name="pupil_ellipse_fits")

    cr_fits = eye_dlc_tracking_data["cr_params"].assign(timestamps=synced_timestamps)
    cr_params = pynwb.core.DynamicTable.from_dataframe(df=cr_fits,
                                                       name="cr_ellipse_fits")

    eye_fits = eye_dlc_tracking_data["eye_params"].assign(timestamps=synced_timestamps)
    eye_params = pynwb.core.DynamicTable.from_dataframe(df=eye_fits,
                                                        name="eye_ellipse_fits")

    eye_tracking_mod.add_data_interface(pupil_params)
    eye_tracking_mod.add_data_interface(cr_params)
    eye_tracking_mod.add_data_interface(eye_params)

    return eye_tracking_mod


def add_eye_gaze_data_interfaces(pynwb_container: pynwb.NWBContainer,
                                 pupil_areas: pd.Series,
                                 eye_areas: pd.Series,
                                 screen_coordinates: pd.DataFrame,
                                 screen_coordinates_spherical: pd.DataFrame,
                                 synced_timestamps: pd.Series) -> pynwb.NWBContainer:

    pupil_area_ts = pynwb.base.TimeSeries(
        name="pupil_area",
        data=pupil_areas.values,
        timestamps=synced_timestamps.values,
        unit="Pixels ^ 2"
    )

    eye_area_ts = pynwb.base.TimeSeries(
        name="eye_area",
        data=eye_areas.values,
        timestamps=synced_timestamps.values,
        unit="Pixels ^ 2"
    )

    screen_coord_ts = pynwb.base.TimeSeries(
        name="screen_coordinates",
        data=screen_coordinates.values,
        timestamps=synced_timestamps.values,
        unit="Centimeters"
    )

    screen_coord_spherical_ts = pynwb.base.TimeSeries(
        name="screen_coordinates_spherical",
        data=screen_coordinates_spherical.values,
        timestamps=synced_timestamps.values,
        unit="Degrees"
    )

    pynwb_container.add_data_interface(pupil_area_ts)
    pynwb_container.add_data_interface(eye_area_ts)
    pynwb_container.add_data_interface(screen_coord_ts)
    pynwb_container.add_data_interface(screen_coord_spherical_ts)

    return pynwb_container


def create_gaze_mapping_nwb_processing_modules(eye_gaze_data: dict):
    # Container for raw gaze mapped data
    raw_gaze_mapping_mod = pynwb.ProcessingModule(name='raw_gaze_mapping',
                                                  description='Gaze mapping processing module raw outputs')

    raw_gaze_mapping_mod = add_eye_gaze_data_interfaces(raw_gaze_mapping_mod,
                                                        pupil_areas=eye_gaze_data["raw_pupil_areas"],
                                                        eye_areas=eye_gaze_data["raw_eye_areas"],
                                                        screen_coordinates=eye_gaze_data["raw_screen_coordinates"],
                                                        screen_coordinates_spherical=eye_gaze_data["raw_screen_coordinates_spherical"],
                                                        synced_timestamps=eye_gaze_data["synced_frame_timestamps"])

    # Container for filtered gaze mapped data
    filt_gaze_mapping_mod = pynwb.ProcessingModule(name='filtered_gaze_mapping',
                                                   description='Gaze mapping processing module filtered outputs')

    filt_gaze_mapping_mod = add_eye_gaze_data_interfaces(filt_gaze_mapping_mod,
                                                         pupil_areas=eye_gaze_data["new_pupil_areas"],
                                                         eye_areas=eye_gaze_data["new_eye_areas"],
                                                         screen_coordinates=eye_gaze_data["new_screen_coordinates"],
                                                         screen_coordinates_spherical=eye_gaze_data["new_screen_coordinates_spherical"],
                                                         synced_timestamps=eye_gaze_data["synced_frame_timestamps"])

    return (raw_gaze_mapping_mod, filt_gaze_mapping_mod)


def add_eye_tracking_ellipse_fit_data_to_nwbfile(nwbfile: pynwb.NWBFile,
                                                 eye_dlc_tracking_data: dict,
                                                 synced_timestamps: pd.Series) -> pynwb.NWBFile:
    eye_tracking_mod = create_eye_tracking_nwb_processing_module(eye_dlc_tracking_data,
                                                                 synced_timestamps)
    nwbfile.add_processing_module(eye_tracking_mod)

    return nwbfile


def add_eye_gaze_mapping_data_to_nwbfile(nwbfile: pynwb.NWBFile,
                                         eye_gaze_data: dict) -> pynwb.NWBFile:
    raw_gaze_mapping_mod, filt_gaze_mapping_mod = create_gaze_mapping_nwb_processing_modules(eye_gaze_data)
    nwbfile.add_processing_module(raw_gaze_mapping_mod)
    nwbfile.add_processing_module(filt_gaze_mapping_mod)

    return nwbfile


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

    running_speed_series = pynwb.base.TimeSeries(
        name=name,
        data=running_speed.values,
        timestamps=running_speed.timestamps,
        unit=unit
    )

    running_mod = ProcessingModule('running', 'Running speed processing module')
    nwbfile.add_processing_module(running_mod)

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
    timestamps_ts = running_mod.get_data_interface('speed').timestamps

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


def create_stimulus_presentation_time_interval(name: str, description: str,
                                               columns_to_add: Iterable) -> pynwb.epoch.TimeIntervals:
    column_descriptions = {
        "stimulus_name": "Name of stimulus",
        "stimulus_block": "Index of contiguous presentations of one stimulus type",
        "temporal_frequency": "Temporal frequency of stimulus",
        "x_position": "Horizontal position of stimulus on screen",
        "y_position": "Vertical position of stimulus on screen",
        "mask": "Shape of mask applied to stimulus",
        "opacity": "Opacity of stimulus",
        "phase": "Phase of grating stimulus",
        "size": "Size of stimulus (see ‘units’ field for units)",
        "units": "Units of stimulus size",
        "stimulus_index": "Index of stimulus type",
        "orientation": "Orientation of stimulus",
        "spatial_frequency": "Spatial frequency of stimulus",
        "frame": "Frame of movie stimulus",
        "contrast": "Contrast of stimulus",
        "Speed": "Speed of moving dot field",
        "Dir": "Direction of stimulus motion",
        "coherence": "Coherence of moving dot field",
        "dotLife": "Longevity of individual dots",
        "dotSize": "Size of individual dots",
        "fieldPos": "Position of moving dot field",
        "fieldShape": "Shape of moving dot field",
        "fieldSize": "Size of moving dot field",
        "nDots": "Number of dots in moving dot field"
    }

    columns_to_ignore = {'start_time', 'stop_time', 'tags', 'timeseries'}

    interval = pynwb.epoch.TimeIntervals(name=name,
                                         description=description)

    for column_name in columns_to_add:
        if column_name not in columns_to_ignore:
            description = column_descriptions.get(column_name, "No description")
            interval.add_column(name=column_name, description=description)

    return interval


def add_stimulus_presentations(nwbfile, stimulus_table, tag='stimulus_time_interval'):
    """Adds a stimulus table (defining stimulus characteristics for each
    time point in a session) to an nwbfile as TimeIntervals.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
    stimulus_table: pd.DataFrame
        Each row corresponds to an interval of time. Columns define the interval
        (start and stop time) and its characteristics.
        Nans in columns with string data will be replaced with the empty strings.
        Required columns are:
            start_time :: the time at which this interval started
            stop_time :: the time  at which this interval ended
    tag : str, optional
        Each interval in an nwb file has one or more tags. This string will be
        applied as a tag to all TimeIntervals created here

    Returns
    -------
    nwbfile : pynwb.NWBFile

    """
    stimulus_table = stimulus_table.copy()
    ts = nwbfile.modules['stimulus'].get_data_interface('timestamps')
    stimulus_names = stimulus_table['stimulus_name'].unique()

    for stim_name in sorted(stimulus_names):
        specific_stimulus_table = stimulus_table[stimulus_table['stimulus_name'] == stim_name]
        # Drop columns where all values in column are NaN
        cleaned_table = specific_stimulus_table.dropna(axis=1, how='all')
        # For columns with mixed strings and NaNs, fill NaNs with 'N/A'
        for colname, series in cleaned_table.items():
            types = set(series.map(type))
            if len(types) > 1 and str in types:
                series.fillna('N/A', inplace=True)
                cleaned_table[colname] = series.transform(str)

        interval_description = (f"Presentation times and stimuli details "
                                f"for '{stim_name}' stimuli")
        presentation_interval = create_stimulus_presentation_time_interval(
            name=f"{stim_name}_presentations",
            description=interval_description,
            columns_to_add=cleaned_table.columns
        )

        for row in cleaned_table.itertuples(index=False):
            row = row._asdict()
            presentation_interval.add_interval(**row, tags=tag, timeseries=ts)

        nwbfile.add_time_intervals(presentation_interval)

    return nwbfile


def add_invalid_times(nwbfile, epochs):
    """
    Write invalid times to nwbfile if epochs are not empty
    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    epochs: list of dicts
        records of invalid epochs

    Returns
    -------
    pynwb.NWBFile
    """
    table = setup_table_for_invalid_times(epochs)

    if not table.empty:
        container = pynwb.epoch.TimeIntervals('invalid_times')

        for index, row in table.iterrows():

            container.add_interval(start_time=row['start_time'],
                                   stop_time=row['stop_time'],
                                   tags=row['tags'],
                                   )

        nwbfile.invalid_times = container

    return nwbfile


def setup_table_for_invalid_times(invalid_epochs):
    """
    Create table with invalid times if invalid_epochs are present

    Parameters
    ----------
    invalid_epochs:  list of dicts
        of invalid epoch records

    Returns
    -------
    pd.DataFrame of invalid times if epochs are not empty, otherwise return None
    """

    if invalid_epochs:
        df = pd.DataFrame.from_dict(invalid_epochs)

        start_time = df['start_time'].values
        stop_time = df['end_time'].values
        tags = [[_type, str(_id), label]
                for _type, _id, label
                in zip(df['type'], df['id'], df['label'])]

        table = pd.DataFrame({'start_time': start_time,
                              'stop_time': stop_time,
                              'tags': tags}
                             )
        table.index.name = 'id'

    else:
        table = pd.DataFrame()

    return table


def setup_table_for_epochs(table, timeseries, tag):

    table = table.copy()
    indices = np.searchsorted(timeseries.timestamps[:], table['start_time'].values)
    if len(indices > 0):
        diffs = np.concatenate([np.diff(indices), [table.shape[0] - indices[-1]]])
    else:
        diffs = []

    table['tags'] = [(tag,)] * table.shape[0]
    table['timeseries'] = [[[indices[ii], diffs[ii], timeseries]] for ii in range(table.shape[0])]
    return table


def add_stimulus_timestamps(nwbfile, stimulus_timestamps, module_name='stimulus'):

    stimulus_ts = TimeSeries(
        data=stimulus_timestamps,
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
        if data.dtype == '<U1':  # data type is composed of unicode characters
            data = trials[c].tolist()
        if not len(data) == len(order):
            if len(data) == 0:
                data = ['']
            nwbfile.add_trial_column(name=c, description=description_dict.get(c, 'NOT IMPLEMENTED: %s' % c), data=data, index=index)
        else:
            nwbfile.add_trial_column(name=c, description=description_dict.get(c, 'NOT IMPLEMENTED: %s' % c), data=data)


def add_licks(nwbfile, licks):

    licks_event_series = TimeSeries(
        data=licks.time.values,
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
    assert rewards_df.index.name == 'timestamps'

    reward_volume_ts = TimeSeries(
        name='volume',
        data=rewards_df.volume.values,
        timestamps=rewards_df.index.values,
        unit='ml'
    )

    autorewarded_ts = TimeSeries(
        name='autorewarded',
        data=rewards_df.autorewarded.values,
        timestamps=reward_volume_ts.timestamps,
        unit=None
    )

    rewards_mod = ProcessingModule('rewards', 'Licking behavior processing module')
    rewards_mod.add_data_interface(reward_volume_ts)
    rewards_mod.add_data_interface(autorewarded_ts)
    nwbfile.add_processing_module(rewards_mod)

    return nwbfile


def add_image(nwbfile, image_data, image_name, module_name, module_description, image_api=None):

    description = '{} image at pixels/cm resolution'.format(image_name)

    if image_api is None:
        image_api = ImageApi

    if isinstance(image_data, sitk.Image):
        data, spacing, unit = ImageApi.deserialize(image_data)
    elif isinstance(image_data, Image):
        data = image_data.data
        spacing = image_data.spacing
        unit = image_data.unit
    else:
        raise ValueError("Not a supported image_data type: {}".format(type(image_data)))

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


def add_segmentation_mask_image(nwbfile, segmentation_mask_image, image_api=None):

    add_image(nwbfile, segmentation_mask_image, 'segmentation_mask_image', 'two_photon_imaging', 'Ophys timestamps processing module', image_api=image_api)


def add_stimulus_index(nwbfile, stimulus_index, nwb_template):

    image_index = IndexSeries(
        name=nwb_template.name,
        data=stimulus_index['image_index'].values,
        unit='None',
        indexed_timeseries=nwb_template,
        timestamps=stimulus_index['start_time'].values)

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


def add_cell_specimen_table(nwbfile, cell_specimen_table):
    cell_roi_table = cell_specimen_table.reset_index().set_index('cell_roi_id')

    # Device:
    device_name = nwbfile.lab_meta_data['metadata'].rig_name
    nwbfile.create_device(device_name,
                          "Allen Brain Observatory")
    device = nwbfile.get_device(device_name)

    # Location:
    location_description = "Area: {}, Depth: {} um".format(
        nwbfile.lab_meta_data['metadata'].targeted_structure,
        nwbfile.lab_meta_data['metadata'].imaging_depth)

    # FOV:
    fov_width = nwbfile.lab_meta_data['metadata'].field_of_view_width
    fov_height = nwbfile.lab_meta_data['metadata'].field_of_view_height
    imaging_plane_description = "{} field of view in {} at depth {} um".format(
        (fov_width, fov_height),
        nwbfile.lab_meta_data['metadata'].targeted_structure,
        nwbfile.lab_meta_data['metadata'].imaging_depth)

    # Optical Channel:
    optical_channel = OpticalChannel(
        name='channel_1',
        description='2P Optical Channel',
        emission_lambda=nwbfile.lab_meta_data['metadata'].emission_lambda)

    # Imaging Plane:
    imaging_plane = nwbfile.create_imaging_plane(
        name='imaging_plane_1',
        optical_channel=optical_channel,
        description=imaging_plane_description,
        device=device,
        excitation_lambda=nwbfile.lab_meta_data['metadata'].excitation_lambda,
        imaging_rate=nwbfile.lab_meta_data['metadata'].ophys_frame_rate,
        indicator=nwbfile.lab_meta_data['metadata'].indicator,
        location=location_description,
        manifold=[],  # Should this be passed in for future support?
        conversion=1.0,
        unit='unknown',  # Should this be passed in for future support?
        reference_frame='unknown')  # Should this be passed in for future support?

    # Image Segmentation:
    image_segmentation = ImageSegmentation(name="image_segmentation")

    if 'two_photon_imaging' not in nwbfile.modules:
        two_photon_imaging_module = ProcessingModule('two_photon_imaging', '2P processing module')
        nwbfile.add_processing_module(two_photon_imaging_module)
    else:
        two_photon_imaging_module = nwbfile.modules['two_photon_imaging']

    two_photon_imaging_module.add_data_interface(image_segmentation)

    # Plane Segmentation:
    plane_segmentation = image_segmentation.create_plane_segmentation(
        name='cell_specimen_table',
        description="Segmented rois",
        imaging_plane=imaging_plane)

    for c in [c for c in cell_roi_table.columns if c not in ['id', 'mask_matrix']]:
        plane_segmentation.add_column(c, c)

    for cell_roi_id, row in cell_roi_table.iterrows():
        sub_mask = np.array(row.pop('image_mask'))
        curr_roi = roi.create_roi_mask(fov_width, fov_height, [(fov_width - 1), 0, (fov_height - 1), 0], roi_mask=sub_mask)
        mask = curr_roi.get_mask_plane()
        csid = row.pop('cell_specimen_id')
        row['cell_specimen_id'] = -1 if csid is None else csid
        row['id'] = cell_roi_id
        plane_segmentation.add_roi(image_mask=mask, **row.to_dict())

    return nwbfile


def add_dff_traces(nwbfile, dff_traces, ophys_timestamps):
    dff_traces = dff_traces.reset_index().set_index('cell_roi_id')[['dff']]

    twop_module = nwbfile.modules['two_photon_imaging']
    data = np.array([dff_traces.loc[cell_roi_id].dff for cell_roi_id in dff_traces.index.values])
    # assert len(ophys_timestamps.timestamps) == len(data)

    cell_specimen_table = nwbfile.modules['two_photon_imaging'].data_interfaces['image_segmentation'].plane_segmentations['cell_specimen_table']
    roi_table_region = cell_specimen_table.create_roi_table_region(
        description="segmented cells labeled by cell_specimen_id",
        region=slice(len(dff_traces)))

    # Create/Add dff modules and interfaces:
    assert dff_traces.index.name == 'cell_roi_id'
    dff_interface = DfOverF(name='dff')
    twop_module.add_data_interface(dff_interface)

    dff_interface.create_roi_response_series(
        name='traces',
        data=data,
        unit='NA',
        rois=roi_table_region,
        timestamps=ophys_timestamps)

    return nwbfile


def add_corrected_fluorescence_traces(nwbfile, corrected_fluorescence_traces):
    corrected_fluorescence_traces = corrected_fluorescence_traces.reset_index().set_index('cell_roi_id')[['corrected_fluorescence']]

    # Create/Add corrected_fluorescence_traces modules and interfaces:
    assert corrected_fluorescence_traces.index.name == 'cell_roi_id'
    twop_module = nwbfile.modules['two_photon_imaging']
    roi_table_region = nwbfile.modules['two_photon_imaging'].data_interfaces['dff'].roi_response_series['traces'].rois
    ophys_timestamps = twop_module.get_data_interface('dff').roi_response_series['traces'].timestamps
    f_interface = Fluorescence(name='corrected_fluorescence')
    twop_module.add_data_interface(f_interface)
    f_interface.create_roi_response_series(
        name='traces',
        data=np.array([corrected_fluorescence_traces.loc[cell_roi_id].corrected_fluorescence for cell_roi_id in corrected_fluorescence_traces.index.values]),
        unit='NA',
        rois=roi_table_region,
        timestamps=ophys_timestamps)

    return nwbfile


def add_motion_correction(nwbfile, motion_correction):

    twop_module = nwbfile.modules['two_photon_imaging']
    ophys_timestamps = twop_module.get_data_interface('dff').roi_response_series['traces'].timestamps

    t1 = TimeSeries(
        name='x',
        data=motion_correction['x'].values,
        timestamps=ophys_timestamps,
        unit='pixels'
    )

    t2 = TimeSeries(
        name='y',
        data=motion_correction['y'].values,
        timestamps=ophys_timestamps,
        unit='pixels'
    )

    motion_module = ProcessingModule('motion_correction', 'Motion Correction processing module')
    motion_module.add_data_interface(t1)
    motion_module.add_data_interface(t2)
    nwbfile.add_processing_module(motion_module)
