#!/usr/bin/python
import sys
import numpy as np
import nwb
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zcorr.nwb"
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Test file for image series")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

borg.set_metadata("optophysiology/camera1/excitation_lambda", "1000 nm") 
borg.set_metadata("optophysiology/camera1/indicator", "GCaMP6s") 
# create different examples of image series
# image can be stored directly, for example by reading a .tif file into
#   memory and storing this data as a byte stream in data[]
# most(all?) examples here will have the time series reference data
#   that is stored externally in the file system

# first, a simple image
orig = borg.create_timeseries("ImageSeries", "source image", "acquisition")
orig.set_description("Pointer to a 640x480 image stored in the file system")
orig.set_value("foo", "CA-Magic, infinite emission spectrum") # need better value
orig.set_value("format", "external")
orig.set_value("external_file", "./foo.png")
orig.set_value("bits_per_pixel", 16)
orig.set_value("dimension", [640, 480])
orig.set_time([0])  # single frame
orig.set_data([0])
orig.finalize()

xy = borg.create_timeseries("TimeSeries", "foo")
xy.set_description("X,Y adjustments to original image necessary for registration")
xy.set_data([1.23, -3.45])
xy.set_time([0])

corr = borg.create_timeseries("ImageSeries", "corrected_image")
corr.set_description("Corrected image")
corr.set_value("foo", "CA-Magic, infinite emission spectrum") # need better value
corr.set_value("format", "external")
corr.set_value("external_file", "./bar.png")
corr.set_value("bits_per_pixel", 16)
corr.set_value("dimension", [640, 480])
corr.set_time([0])  # single frame
corr.set_data([0])

mod = borg.create_module("pipeline")
iface = mod.create_interface("MotionCorrection")
iface.add_corrected_image("2photon", orig, xy, corr)

# create a new interface to check linking logic
# the timeseries xy and corr were finalized when first added to the iface.
#   the API should detect this and create links to the existing series
iface.add_corrected_image("4photon", orig.full_path(), xy.full_path(), corr)
iface.add_corrected_image("6photon", orig, xy, corr.full_path())

iface.finalize()


mod.finalize()


# when all data is entered, close the Borg file
borg.close()

