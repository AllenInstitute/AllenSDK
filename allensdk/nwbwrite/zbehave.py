#!/usr/bin/python
import sys
import nwb
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zbehav.nwb"
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Test file for behavioral interfaces")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

mod = borg.create_module("simple behavioral module")
mod.set_description("module description")

iface_1 = mod.create_interface("BehavioralEpochs")
iface_1.set_source("imagination")

# create an IntervalSeries
# This will normally be stored in processing or stimulus
# For this example, store it in stimulus (keeps things more simple)
interval = borg.create_timeseries("IntervalSeries", "intervals", "other")
interval.set_description("Sample interval series -- two series are overlaid here, one with a code '1' and another with the code '2'")
interval.set_comment("For example, '1' represents sound on/off and '2' represents light on/off")
evts = [ 1, -1, 2, -2, 1, -1, 2, 1, -1, -2, 1, 2, -1, -2 ]
# note: some timestamps will be duplicated if two different events start 
#   and/or stop at the same time
t = [ 1, 2, 2, 3, 5, 6, 6, 7, 8, 8, 10, 10, 11, 15 ]
interval.set_data(evts)
interval.set_time(t)
iface_1.add_timeseries(interval)
interval.finalize()
iface_1.finalize()

iface_2 = mod.create_interface("BehavioralEvents")
iface_2.set_source("active imagination")

# create an AnnotationSeries
# This will be sotred in 'acquisiiton' as annotations are an
#   observation or a record of something else that happened 
#   (i.e., the Annotation didn't change the experimental environment)
annot = borg.create_timeseries("AnnotationSeries", "notes", "acquisition")
# create dummy entries at Fibonacci times
prev1 = -1.0
prev2 = -1.0
for i in range(10):
    t = 1.0
    if prev1 < 0:
        prev1 = 1.0
    elif prev2 < 0:
        prev2 = 1.0
    else:
        t = prev1 + prev2
        prev2 = prev1
        prev1 = t
    annot.add_annotation("dummy entry %d" % i, t)

# add a description
annot.set_description("This is an AnnotationSeries with sample data")
annot.set_comment("The comment and description fields can store arbitrary human-readable data")
annot.set_source("Source of data is the sample file " + __file__)
annot.finalize()

iface_2.add_timeseries_as_link("notes", annot.full_path())
# finalize the time series
iface_2.finalize()

mod.finalize()

# when all data is entered, close the Borg file
borg.close()

