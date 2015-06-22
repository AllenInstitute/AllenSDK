#!/usr/bin/python
import sys
import sys
import nwb
import numpy as np

# create a new borg file. If we don't specify a time, the present time
#   will be used as the experiment start time
borg_file = nwb.NWB(filename="sample_patchclamp.nwb", identifier="test", overwrite=True, description="Test patch clamp representation")

# this file stores data from two virtual patch clamp recordings, one
#   current clamp and one voltage clamp. Each recording is split into
#   stimulus and response elements

########################################################################
########################################################################
# METADATA -- this is an example of setting experimental metadata for 
#   an intracellular ephys recording
#
# skip down to DATA for the meat and potatoes of the script
# 
# Recording hardware
# define the device (hardware) that we're using to record
dev_name = "Recording device"
borg_file.set_metadata("devices/"+dev_name, "<Info about device (make, model, characteristics, etc)>")
# Slice info
borg_file.set_metadata("slices", "<Info about slice, or slices, including bath solution, thickness, etc>")
# Electrode info
# define the electrode that we're recording from, including 
#   info about recording device
trode_name = "electrode-1"
trode_path = "intracellular_ephys/" + trode_name + "/"
borg_file.set_metadata(trode_path+"description", "<Description of electrode>")
borg_file.set_metadata(trode_path+"location", "<Location info (eg, reference to slice)>")
borg_file.set_metadata(trode_path+"device", dev_name)
borg_file.set_metadata(trode_path+"initial_access_resistance", "<e.g., 12.345 MOhm>")
borg_file.set_metadata(trode_path+"electrode_resistance", "<e.g., 1-3 MOhm>")
borg_file.set_metadata(trode_path+"filtering", "<e.g., Bessel filter, 10KHz cutoff, params=...>")
borg_file.set_metadata(trode_path+"seal", 7.7)

########################################################################
########################################################################
# DATA -- create time series to store the data

##########################################
# define virtual data

# voltage clamp stimulus, response and time
vc_stim = np.arange(1000)
vc_resp = np.arange(1000)
vc_t = np.arange(1000) * 0.001

# current clamp stimulus, response and time
cc_stim = np.arange(1000)
cc_resp = np.arange(1000)
cc_t = 5 + np.arange(1000) * 0.001

##########################################
# voltage clamp 
# stimulus
stim = borg_file.create_timeseries("VoltageClampStimulusSeries", "voltage-clamp", "stimulus")
stim.set_description("Example creating a voltage clamp stimulus. Data values are bogus")
stim.set_value("electrode_name", trode_name)
# stimulus is in units of volts. The '1' is the conversion between 
#   'data' and 'Volts'.
# the 1e-5 is the resolution of data -- ie, the +/- accuracy range 
#   about each data element
stim.set_data(vc_stim, units="Volts", conversion=1, resolution=1e-5) 
stim.set_time(vc_t)
# finilize the time series when we're done adding data to it
stim.finalize()
# recording
rec = borg_file.create_timeseries("VoltageClampSeries", "voltage-clamp", "acquisition")
rec.set_description("Example creating a current clamp time series. Data values are bogus")
rec.set_value("electrode_name", trode_name)
rec.set_data(vc_resp, units="Amps", conversion=1, resolution=1e-5)
rec.set_time(vc_t)
rec.set_value("capacitance_fast", 1e-6)
rec.set_value("capacitance_slow", 2e-6)
rec.set_value("resistance_comp_bandwidth", 3e6)
rec.set_value("resistance_comp_prediction", 1e6)
rec.set_value("resistance_comp_correction", 2e6)
rec.set_value("whole_cell_series_resistance_comp", 50)
rec.set_value("whole_cell_capacitance_comp", 5e9)
rec.set_value("gain", 10e3)
rec.finalize()

##########################################
# current clamp 
# stimulus
stim = borg_file.create_timeseries("CurrentClampStimulusSeries", "current-clamp", "stimulus")
stim.set_description("Example creating a current clamp stimulus. Data values are bogus")
stim.set_value("electrode_name", trode_name)
stim.set_data(cc_stim, "Amps", 1, 1e-5) 
stim.set_time(cc_t)
# finilize the time series when we're done adding data to it
stim.finalize()
# recording
rec = borg_file.create_timeseries("CurrentClampSeries", "current-clamp", "acquisition")
rec.set_description("Example creating a current clamp time series. Data values are bogus")
rec.set_data(cc_resp, "Volts", 1, 1e-5)
rec.set_time(cc_t)
rec.set_value("electrode_name", trode_name)
rec.set_value("bias_current", 50e-9)
rec.set_value("capacitance_compensation", 5.5)
rec.set_value("resistance_compensation", 6.6)
rec.set_value("access_resistance", 3.3)
rec.set_value("bridge_balance", 1.9)
rec.set_value("gain", 8.8)
rec.finalize()

# close the file when we're done. the file won't be usable otherwise
borg_file.close()

