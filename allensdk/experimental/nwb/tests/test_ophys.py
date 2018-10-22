from allensdk.experimental.nwb import ophys
from collections import Iterable
from pynwb import NWBFile, NWBHDF5IO
from pynwb.device import Device
from pynwb.ophys import OpticalChannel, ImageSegmentation, DfOverF
import numpy as np
import datetime
import pytest


@pytest.fixture
def nwb_filename(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test").join("test.nwb")
    return str(nwb)


@pytest.fixture
def roi_mask_dict():
    rois = {1: np.ones((5,5), dtype=bool),
            2: np.zeros((5,5), dtype=bool)}

    return rois


@pytest.fixture
def dff():
    return np.vstack([np.arange(100), np.arange(100)/2])


@pytest.fixture
def timestamps():
    return np.arange(100)


@pytest.fixture(scope='function')
def base_ophys_nwbfile(nwbfile):
    device = Device('death star laser', source='a source')
    nwbfile.add_device(device)

    ophys_module = nwbfile.create_processing_module(
        name='ophys_module',
        source='sources sources everywhere',
        description='Processing module for 2P calcium responses')

    dff_interface = DfOverF(name='dff_interface',
                            source="imaging plane extraction")

    ophys_module.add_data_interface(dff_interface)

    return nwbfile


@pytest.fixture(scope='function')
def imaging_plane_nwb(base_ophys_nwbfile):
    nwbfile = base_ophys_nwbfile

    optical_channel = OpticalChannel(
        name='optical_channel',
        source='data source',
        description='2P Optical Channel',
        emission_lambda=520.)

    nwbfile.create_imaging_plane(
        name='imaging_plane',
        source='a source',
        optical_channel=optical_channel,
        description='Imaging plane ',
        device=nwbfile.get_device('death star laser'),
        excitation_lambda=920.,
        imaging_rate="31.",
        indicator='GCaMP6f',
        location='VISp',
        manifold=[],
        conversion=1.0,
        unit='unknown',
        reference_frame='unknown')

    return nwbfile


@pytest.fixture(scope='function')
def image_segmentation_nwb(imaging_plane_nwb):
    nwbfile = imaging_plane_nwb

    image_segmentation = ImageSegmentation(
        name="image_segmentation",
        source="image source")
    
    ophys_module = nwbfile.get_processing_module("ophys_module")

    ophys_module.add_data_interface(image_segmentation)

    return nwbfile


def test_basic_roundtrip(base_ophys_nwbfile, nwb_filename):
    nwb_out = base_ophys_nwbfile

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwb_out)

    nwb_in = NWBHDF5IO(nwb_filename, mode='r').read()

def test_get_image_segmentation(base_ophys_nwbfile, nwb_filename):
    ophys_module = base_ophys_nwbfile.get_processing_module('ophys_module')
    seg1 = ophys.get_image_segmentation(ophys_module, "source", "test_seg")
    seg2 = ophys.get_image_segmentation(ophys_module, "source")
    seg3 = ophys.get_image_segmentation(ophys_module, "source")
    assert(seg1 != seg2 and seg1 != seg3)
    assert(seg2 == seg3)


def test_get_plane_segmentation(image_segmentation_nwb, roi_mask_dict):
    imaging_plane = image_segmentation_nwb.get_imaging_plane('imaging_plane')
    ophys_module = image_segmentation_nwb.get_processing_module('ophys_module')
    image_segmentation = ophys.get_image_segmentation(ophys_module, "source")
    seg1 = ophys.get_plane_segmentation(image_segmentation, imaging_plane,
                                        roi_mask_dict, "source", "description",
                                        name="test_seg")
    seg2 = ophys.get_plane_segmentation(image_segmentation, imaging_plane,
                                        roi_mask_dict, "source", "description")
    seg3 = ophys.get_plane_segmentation(image_segmentation, imaging_plane,
                                        roi_mask_dict, "source", "description")
    assert(seg1 != seg2 and seg1 != seg3)
    assert(seg2 == seg3)


def test_get_dff_series(image_segmentation_nwb, roi_mask_dict, dff, timestamps):
    # TODO: make this a proper unit test by fixturing nwb with plane segmentation
    imaging_plane = image_segmentation_nwb.get_imaging_plane('imaging_plane')
    ophys_module = image_segmentation_nwb.get_processing_module('ophys_module')
    image_segmentation = ophys.get_image_segmentation(ophys_module, "source")
    plane_segmentation = ophys.get_plane_segmentation(image_segmentation,
                                                      imaging_plane,
                                                      roi_mask_dict,
                                                      "source",
                                                      "description")
    dff_interface = ophys_module.get_data_interface('dff_interface')
    rt = plane_segmentation.create_roi_table_region(
        description='Segmented cells with cell_specimen_ids.',
        region=slice(len(roi_mask_dict.keys())))
    series1 = ophys.get_dff_series(dff_interface, rt, dff, timestamps,
                                   "source", name="test")
    series2 = ophys.get_dff_series(dff_interface, rt, dff, timestamps,
                                   "source")
    series3 = ophys.get_dff_series(dff_interface, rt, dff, timestamps,
                                   "source")