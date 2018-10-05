import datetime
import h5py
from .timestamps import get_timestamps_from_sync
from pynwb import NWBFile
from pynwb.ophys import DfOverF, ImageSegmentation
from pynwb.form.backends.hdf5.h5_utils import H5DataIO


def create_base_nwb(session_metadata):
    nwbfile = NWBFile(
        source='Allen Brain Observatory',
        session_description=session_metadata["session_description"],
        institute='The Allen Institute for Brain Science',
        identifier=session_metadata["experiment_id"],
        session_id=session_metadata["session_id"],
        experiment_id=session_metadata["experiment_id"],
        session_start_time=session_metadata["session_start_time"],
        file_create_date=datetime.datetime.now()
    )

    nwbfile.create_device(session_metadata["device_name"])

    return nwbfile


def add_ophys_module(nwbfile, name, description, dff_source):
    module = nwbfile.create_processing_module(
        name=name,
        source='Allen Brain Observatory',
        description=description)

    dff_interface = DfOverF(
        name='dff_interface',
        source=dff_source)

    module.add_data_interface(dff_interface)

    return module


def add_imaging_plane(nwbfile, optical_channel, session_metadata,
                      description, name='imaging_plane_1'):

    device = nwbfile.get_device(session_metadata['device_name'])

    location = "Area: {}, Depth: {} um".format(
        session_metadata['targeted_structure'],
        session_metadata['imaging_depth_um'])

    imaging_plane = nwbfile.create_imaging_plane(
        name=name,
        source='a source',
        optical_channel=optical_channel,
        description=description,
        device=device,
        excitation_lambda=session_metadata['excitation_lambda'],
        imaging_rate=session_metadata['imaging_rate'],
        indicator=session_metadata['indicator'],
        location=location,
        manifold=[], # Should this be passed in for future support?
        conversion=1.0,
        unit='unknown', # Should this be passed in for future support?
        reference_frame='unknown') # Should this be passed in for future support?

    return imaging_plane


def get_image_segmentation(ophys_module,
                           source,
                           name="image_segmentation"):
    """Get an image segmentation by name, create it if it doesn't exist"""
    image_segmentation = ophys_module.data_interfaces.get(name, None)

    if image_segmentation is None:
        image_segmentation = ImageSegmentation(
            name=name,
            source=source)
        ophys_module.add_data_interface(image_segmentation)
    
    return image_segmentation


def get_plane_segmentation(image_segmentation, imaging_plane, roi_mask_dict,
                           source, description, name="plane_segmentation"):
    """Get a plane segmentation by name, create it if it doesn't exist"""
    plane_segmentation = image_segmentation.plane_segmentations.get(name, None)

    if plane_segmentation is None:
        plane_segmentation = image_segmentation.create_plane_segmentation(
            name=name,
            description=description,
            source=source,
            imaging_plane=imaging_plane)

        for roi_id, roi_mask in roi_mask_dict.items():
            plane_segmentation.add_roi(str(roi_id), [], roi_mask)

    return plane_segmentation


def get_dff_series(dff_interface, roi_table_region, dff, timestamps,
                   source, name='df_over_f', **compression_opts):
    dff_series = dff_interface.roi_response_series.get(name, None)

    if dff_series is None:
        dff_series = dff_interface.create_roi_response_series(
            name='df_over_f',
            source=source,
            data=H5DataIO(dff, **compression_opts),
            unit='NA',
            rois=roi_table_region,
            timestamps=timestamps)

    return dff_series


class OphysAdapter(object):
    def __init__(self, experiment_id, api, ophys_key="2p_vsync",
                 use_falling_edges=True):
        self.experiment_id = experiment_id
        self.sync_file = api.get_sync_file()
        self.ophys_key = ophys_key
        self.use_falling_edges = use_falling_edges
        self._api = api
        self._dff_table = None
        self._metadata = None
        self._channels = {}
        self._timestamps = None

    @property
    def dff_source(self):
        return self._api.get_dff_file(self.experiment_id)

    @property
    def dff_traces(self):
        with h5py.File(self.dff_source, "r") as f:
            traces = f["data"].value

        timestamps = self.timestamps

        if len(timestamps) > traces.shape[1]:
            timestamps = timestamps[:traces.shape[1]]

        return traces, timestamps

    @property
    def timestamps(self):
        if self._timestamps is None:
            self._timestamps = get_timestamps_from_sync(
                self.sync_file, self.ophys_key, self.use_falling_edges)

        return self._timestamps

    @property
    def roi_mask_dict(self):
        return self._api.roi_mask_dict(self.experiment_id)

    @property
    def session_metadata(self):
        if self._metadata is None:
            self._metadata = self._api.session_metadata(self.experiment_id)

        return self._metadata

    def get_optical_channel(self, channel='channel_1'):
        metadata = self.session_metadata()

        optical_channel = self._channels.get(channel, None)
        if optical_channel is None:
            optical_channel = OpticalChannel(
                name=channel,
                source=metadata['device_name'],
                description='2P Optical Channel',
                emission_lambda=metadata['emission_lambda'])

            self._channels[channel] = optical_channel

        return optical_channel
