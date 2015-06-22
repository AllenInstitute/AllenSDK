#!/usr/bin/python
import sys
import numpy as np
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_spatial_series.nwb", identifier="test", overwrite=True, description="test for spatial time series")

# create an SpatialSeries
# This will normally be stored in processing because it is derived data
#   (eg, after analysis of video tracking) but it can be raw aquisition
#   data (eg, streaming from GPS)
# For this example, store it in acquisition (keeps things more simple)
spatial = borg.create_timeseries("SpatialSeries", "position", "acquisition")
spatial.set_description("Sample spatial series -- stores X,Y coordinates")
spatial.set_value("reference_frame", "Coordinates are in centimeters as measured by video tracking system. 'Up' is the blue side of the enclosure. Origin (0,0) is the orange dot in the enclosure")
# generate some bogus data
x = np.cos(np.arange(1000))
y = np.sin(np.arange(1000))
pts = np.transpose([ x, y])
# coordinates are reported in centimeters, but storage always uses SI units
# we need to set the conversion value (param #3) to 0.01, to indicate the
#   how much we need to multiply data[] values by to achieve the stated
#   SI unit
spatial.set_data(pts, units="meters", conversion=0.01, resolution=0.001)
t = np.arange(1000) * 0.001
spatial.set_time(t)
# 
spatial.finalize()

# when all data is entered, close the Borg file
borg.close()

