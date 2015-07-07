#!/usr/bin/python
import sys
import nwb
from nwbco import *
fname = "sample_ziface.nwb"

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename=fname, identifier="test", overwrite=True, description="Test file for behavioral interfaces")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

mod = borg.create_module("spatial module")
mod.set_description("module description")

iface_pos = mod.create_interface("Position")
iface_pos.set_source("imagination")
pos_ts = borg.create_timeseries("SpatialSeries", "pos", "other")
data = [[1, 2], [1, 3], [6, 6]]
t = [ 1.0, 1.1, 1.3 ]
pos_ts.set_data(data)
pos_ts.set_time(t)
pos_ts.set_value("reference_frame", "top is north, center is center of universe")
iface_pos.add_timeseries(pos_ts)
pos_path = pos_ts.full_path()
iface_pos.finalize()    # implicit call to pos_ts.finalize()

iface_eye = mod.create_interface("EyeTracking")
iface_eye.set_source("active imagination")
eye_ts = borg.create_timeseries("SpatialSeries", "eye", "other")
eye_ts.set_path(iface_eye.full_path())
eye_ts.set_data_as_link(pos_path)
eye_ts.set_time_as_link(pos_path)
eye_ts.set_value("num_samples", len(data))
iface_eye.add_timeseries(eye_ts)
iface_eye.finalize()

mod.finalize()

# ------------------

mod = borg.create_module("electrical module")
mod.set_description("another module description")
iface_elec = mod.create_interface("FilteredEphys")
iface_elec.set_source("dreams")

borg.set_metadata("extracellular_ephys/electrode_map", [0.4, 0.1])
borg.set_metadata("extracellular_ephys/electrode_group", "trode-1")
borg.set_metadata("extracellular_ephys/impedance", 0.3e6)
borg.set_metadata("extracellular_ephys/trode-1/location", "top of skull")

elec = borg.create_timeseries("ElectricalSeries", "trode")
data = [ 0.071, 0.072, 0.069, 0.070, 0.067 ]
elec.set_data(data)
elec.set_time_by_rate(0.3, 0.1)
elec.set_value("electrode_idx", 0)
elec.set_value("num_samples", len(data))
iface_elec.add_timeseries(elec)
iface_elec.finalize()

mod.finalize()

# when all data is entered, close the Borg file
borg.close()

