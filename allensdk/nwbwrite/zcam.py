#!/usr/bin/python
import sys
import nwb
import numpy as np

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_file.nwb"
print "Creating " + fname
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Script to demo some CAM requirements")

mod = borg.create_module("Eye tracking")    # module name is arbitrary
mod.set_description("Module to track pupil size and gaze direction")

# interface to store gaze direction. the interface provides a wrapper
#   for and a label for software reading the data to know what to expect
iface_eye = mod.create_interface("EyeTracking")
iface_eye.set_source("<Name and path of video file tracking eye>")
# gaze itself is stored in a space-representing time series called a
#   'SpatialSeries'
eye_ts = borg.create_timeseries("SpatialSeries", "eye")
# generate some bogus data to store
# time is a series of floats (timestamps)
# data is a series of x,y positions
t = np.arange(10) * 0.001
data = [[5,5], [5,6], [5,7], [6,6], [7,5], [8,3], [6,2], [4,4], [3,5], [4,5]]
eye_ts.set_data(data)
eye_ts.set_time(t)
iface_eye.add_timeseries(eye_ts)    # implicitly calls eye_ts.finalize()
# remember where this time series is stored, so we can link to it later
eye_ts_path = eye_ts.full_path()
iface_eye.finalize()

# create interface to store pupil size
iface_pup = mod.create_interface("PupilTracking")
# pupil data is stored in a regulat time series
pup_ts = borg.create_timeseries("TimeSeries", "pupil size")
# assume eye and pupil tracking are on the same clock so that we can
#   re-use the time data (this is also an example of using an HDF5 link
#   so two time series can reference the same data, avoiding duplication)
# annoying hack: to use link we need to specify where timeseries will
#   be stored, as links are tracked by the kernel. the path where the
#   time series is stored is the path of the interface
pup_ts.set_path(iface_pup.full_path())
pup_ts.set_time_as_link(eye_ts_path)
sz = [0.5, 0.5, 0.5, 0.4, 0.5, 0.55, 0.6, 0.5, 0.45, 0.5 ]
pup_ts.set_data(sz);
# when links are used for time or data, we need to specify how many
#   samples are relevant
pup_ts.set_value("num_samples", len(sz))
pup_ts.set_description("Pupil size data")
iface_pup.set_source("<Name and path to video file tracking eye>")
iface_pup.add_timeseries(pup_ts)
iface_pup.finalize()

# finish up the eye-tracking module. generally speaking, all objects
#   need to be 'finalized' when done, so their data is written to disk
#   and memory resources can be freed
mod.finalize()

# ------------------

# create a module to store image segmentation and df/f data
mod = borg.create_module("2-photon")    # module name is arbitrary
mod.set_description("Image segmentation and dF/F")
iface_seg = mod.create_interface("ImageSegmentation")
iface_seg.set_source("<name of module or time series where data here originated from, or other pointer to source>")
# assume 2-photon image is 512x512
# file can store data from multiple imaging planes or fields of view
# we need to specify one
plane = "imaging_plane_1"
iface_seg.create_imaging_plane(plane, "<description of imaging plane>")
for i in range(10):
    # ROI name is arbitrary, but it must be consistent between places it
    #   is used/referenced (eg, here plus dF/F)
    roi_name = "roi-%d" % i 
    # create pixel list that defines ROI (one can supply 2D matrix instead)
    pixels = []
    # create bogus mask
    for j in range(10):
        pixels.append([j+100+i, j+100])
    # specify weight of pixels
    wt = np.zeros(len(pixels)) + 1.0
    iface_seg.add_roi_mask_pixels(plane, roi_name, "<description of ROI>", pixels, wt, 512, 512)
iface_seg.finalize()

# create sample dF/F data
iface_dff = mod.create_interface("DfOverF")
iface_dff.set_source("This module's ImageSegmentation interface")
# dF/F is stored in a matrix, with each row storing the signal from one ROI
# data is stored in a time series 'RoiResponseSeries'. this is a normal
#   time series with the additional requirement that it stores an array
#   of ROI names, one for each row, and a field indicating the segmentation
#   source
# sample ROI data
data = np.zeros((10,100))
t = np.arange(100) * 0.001
# create array of ROI names for each matrix row
roi_names = []
for i in range(10):
    roi_name = "roi-%d" % i 
    roi_names.append(roi_name)
ts = borg.create_timeseries("RoiResponseSeries", plane)
ts.set_description("dF/F data for imaging plane " + plane)
ts.set_data(data)
ts.set_time(t)
ts.set_value("roi_names", roi_names)
ts.set_value("segmentation_source", "This module's ImageSegmentation interface")
iface_dff.add_timeseries(ts)
iface_dff.finalize()

mod.finalize()

# when all data is entered, close the Borg file
borg.close()

