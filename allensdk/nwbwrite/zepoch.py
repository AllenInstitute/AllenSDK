#!/usr/bin/python
import sys
import nwb
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zepoch.nwb"
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Test file for epochs")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

# create an IntervalSeries
# This will normally be stored in processing or stimulus
# For this example, store it in stimulus (keeps things more simple)
interval = borg.create_timeseries("IntervalSeries", "intervals", "stimulus")
interval.set_description("Sample interval series -- two series are overlaid here, one with a code '1' and another with the code '2'")
interval.set_comment("For example, '1' represents sound on/off and '2' represents light on/off")
evts = [ 1, -1, 2, -2, 1, -1, 2, 1, -1, -2, 1, 2, -1, -2 ]
# note: some timestamps will be duplicated if two different events start 
#   and/or stop at the same time
t = [ 1, 2, 2, 3, 5, 6, 6, 7, 8, 8, 10, 10, 11, 15 ]
interval.set_data(evts)
interval.set_time(t)
# 
interval.finalize()

ep1 = borg.create_epoch("Epoch 1", 2.5, 8.5)
ep1.add_tag("foo")
ep1.add_tag("foo1")
ep1.add_timeseries("interval-series", "stimulus/presentation/intervals")

ep2 = borg.create_epoch("Epoch 2", 6.5, 11.5)
ep2.add_tag("foo")
ep2.add_tag("bar")
ep2.set_value("custom_field", "nonesense")
ep2.add_ignore_interval(1, 10)
ep2.add_ignore_interval(10.5, 10.75)
ep2.add_timeseries("interval-series", "stimulus/presentation/intervals")

# when all data is entered, close the Borg file
borg.close()

