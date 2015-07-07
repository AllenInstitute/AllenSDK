#!/usr/bin/python
import sys
import nwb
import numpy as np
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zelectrical.nwb"
borg_file = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Electrical series test script; timeseries hard link between single and double")
print "Creating " + fname
borg_file.set_metadata_from_file("source_script", sys.argv[0])

# create two electrical series, one with a single electrode and one with many
# then create a spike event series

# first create the electrode map
electrode_map = [[0, 0, 0], [0, 1, 0], [0, 0, 10], [0, 2, 10]]
electrode_group = [ "p0", "p0", "p1", "p1" ]
borg_file.set_metadata("extracellular_ephys/electrode_map", electrode_map)
borg_file.set_metadata("extracellular_ephys/electrode_group", electrode_group)
# set electrode impedances
borg_file.set_metadata("extracellular_ephys/impedance", [ 1e6, 1.1e6, 1.2e6, 1.3e6 ])

# define some bogus data to store
data = np.arange(1000)
timestamps = np.arange(1000) * 0.001

# create time series with single electrode
single = borg_file.create_timeseries("ElectricalSeries", "mono", "acquisition")
single.set_comment("Data corresponds to a single electrode")
single.set_data(data, units="n/a", conversion=1, resolution=1, dtype='f4')
single.set_time(timestamps)
single.set_value("electrode_idx", 1)
single.finalize()

# create time series with two electrodes
# link time to single
double = borg_file.create_timeseries("ElectricalSeries", "duo", "acquisition")
double.set_comment("Data corresponds to two electrodes")
double.set_data(data, units="n/a", conversion=1, resolution=1, dtype='f4')
double.set_time_as_link(single)
double.set_value("num_samples", len(timestamps))
double.set_value("electrode_idx", [0, 1])
double.finalize()

spike = borg_file.create_timeseries("SpikeEventSeries", "spike", "acquisition")
spike.set_comment("Snapshots of events pulled from a recording")
spike.set_value("electrode_idx", [2, 3])
spike.set_value("number_samples", 8) # normally >20
spike.set_value("source", "Data from device FooBar-X1 using dynamic multi-phasic threshold of 5xRMS")
# make some bogus simulated data
evt = np.zeros((8,2))
evt[3][0] = 1.0
evt[4][0] = -0.5
evt[3][1] = 0.5
evt[4][1] = -0.25
data = []
t = []
last = 1.0
for i in range(20):
    data.append(evt)
    last = last + (i * 17) % 29
    t.append(last)
spike.set_data(data, units="Volts", conversion=0.001, resolution=1, dtype='f4')
spike.set_time(t)
spike.finalize()

# close file, otherwise it will fail to write properly
borg_file.close()

