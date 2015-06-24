#!/usr/bin/python
import sys
import numpy as np
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_image_series.nwb", identifier="test", overwrite=True, description="Test file for image series")

borg.set_metadata("optophysiology/camera1/excitation_lambda", "1000 nm") 
borg.set_metadata("optophysiology/camera1/indicator", "GCaMP6s") 
# create different examples of image series
# image can be stored directly, for example by reading a .tif file into
#   memory and storing this data as a byte stream in data[]
# most(all?) examples here will have the time series reference data
#   that is stored externally in the file system

# first, a simple image
simple = borg.create_timeseries("ImageSeries", "image_series", "acquisition")
simple.set_description("Pointer to a 640x480 image stored in the file system")
simple.set_value("foo", "CA-Magic, infinite emission spectrum") # need better value
simple.set_value("format", "external")
simple.set_value("external_file", "./foo.png")
simple.set_value("bits_per_pixel", 16)
simple.set_value("dimension", [640, 480])
simple.set_time([0])  # single frame
simple.set_data([0])
simple.finalize()

# now an image stack, such as a stimulus movie
stim = borg.create_timeseries("OpticalSeries", "white-noise", "stimulus")
stim.set_description("Example of video stimulus (eg, white-noise movie) presented during an experiment")
# must declare all fields from the ImageSeries
stim.set_description("Pointer to a 640x480 movie stored in the file system")
stim.set_value("format", "external")
stim.set_value("external_file", "./bar.mp4")
stim.set_value("bits_per_pixel", 8)
stim.set_value("dimension", [640, 480])
# declare OpticalSeries fields too (fields can be declared in any order)
stim.set_value("field_of_view", [0.3, 0.15]) # in meters
stim.set_value("distance", 0.5) # in meters
stim.set_value("orientation", "Top of video monitor is up")
# no need to set_data() for externally stored file
stim.set_time(np.arange(1000) * 0.030) # 1000 frames w/ 30ms interval
stim.set_data([0])
stim.finalize()

# now a 2P image stack
twop = borg.create_timeseries("TwoPhotonSeries", "2P series", "acquisition")
# all fields from OpticalSeries
twop.set_description("Pointer to a 512x512 image stack in the file system")
twop.set_value("format", 'external')
twop.set_value("external_file", "./foobar.avi")
twop.set_value("bits_per_pixel", 8)
twop.set_value("dimension", [512, 512])
# all fields from ImageSeries
twop.set_value("field_of_view", [0.100, 0.100]) # in meters
twop.set_value("distance", 0.1) # in meters
twop.set_value("orientation", "Top of frame is medial boundary of M1; rostral to right")
# fields from TwoPhotonSeries
twop.set_value("pmt_gain", 100) # need better value
# use some obsoleted values to test adding custom fields
twop.set_value("wavelength", 8.3e-9) # need better value
twop.set_value("indicator", "CA-Magic, infinite emission spectrum") # need better value
twop.set_value("imaging_depth", 0.0003)
twop.set_value("scan_line_rate", 39.0125)
# no need to set_data() for externally stored file
twop.set_time(np.arange(1000) * 0.030) # 1000 frames w/ 30ms interval
twop.set_data([0])
twop.finalize()


# when all data is entered, close the Borg file
borg.close()

