#!/usr/bin/python
import sys
import nwb
import numpy as np
from nwbco import *
fname = "sample_zsam.nwb"

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Script to demo some other CAM requirements")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

# 2 types of stimuli
# one is orientation gratings
# two is frames presented from a image library

# orientation gratings
ori = borg.create_timeseries("AbstractFeatureSeries", "video-stim", "stimulus")
ori.set_description("Stimulus series for drifting gratings")
features = ["orientation", "spatial frequency", "temporal frequency", "contrast" ]
ori.set_value("features", features)
ori.set_value("feature_units", ["degrees", "Hertz", "Hertz", "percent"])
# create some sample data
t = np.arange(12) * 0.001
data = [
    [ 120.0,    0.25,    0.5,    0.25 ],
    [ 180.0,    0.25,    0.5,    0.25 ],
    [ 240.0,    0.25,    0.5,    0.25 ],
    [ 300.0,    0.25,    0.5,    0.25 ],
    [ 120.0,    0.5,    0.5,    0.25 ],
    [ 180.0,    0.5,    0.5,    0.25 ],
    [ 240.0,    0.5,    0.5,    0.25 ],
    [ 300.0,    0.5,    0.5,    0.25 ],
    [ 120.0,    0.5,    0.25,    0.25 ],
    [ 180.0,    0.5,    0.25,    0.25 ],
    [ 240.0,    0.5,    0.25,    0.25 ],
    [ 300.0,    0.5,    0.25,    0.25 ]
]
ori.set_data(data)
ori.set_time(t)
ori.finalize()

# frames from a library
# since we don't have an actual library to reference, we need to make one
lib = borg.create_timeseries("ImageSeries", "image library", "template")
lib.set_description("Stack of discrete images")
lib.set_value("format", "tif")
lib.set_value("external_file", "/data/cam/image_library/foo_bar.tif")
lib.set_value("dimension", [ 512, 512] )
# time is irrelevant for templates but we still must provide it (w/
#   present API)
lib.set_time([0])   
# data is irrelevant when an external file is referenced, but we still must
#   provide it (w/ present API)
lib.set_data([0])
lib.finalize()

# 
frames = borg.create_timeseries("IndexSeries", "frame-stim", "stimulus")
frames.set_description("Images presented during experiment")
# this should be a link, but to play nice w/ SLAPI, it's a path
frames.set_value_as_link("base_timeseries", lib)
# create sample data
frame_num = 10 * np.arange(10)
frames.set_data(frame_num)
frames.set_time(t)
frames.finalize()
    

# when all data is entered, close the Borg file
borg.close()

