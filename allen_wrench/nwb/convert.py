#!/usr/bin/python
import h5py
import datetime
import copy
import time
import sys
import numpy as np
import sequence
import math

TESTING = True
TEST_N = 5

if len(sys.argv) != 3:
	if TESTING:
		infile = "154747.h5"
		outfile = "foo.h5"
	else:
		print "Usage: %s <input h5> <output nwb>" % sys.argv[0]
		sys.exit(1)
else:
	infile = sys.argv[1]
	outfile = sys.argv[2]

FILE_VERSION_STR = "0.0.1"

IDENT_PREFIX = "Allen Institute, NDB-dev: "

########################################################################
########################################################################

# returns date string based on number of seconds since midnight Jan 1st, 1904
def timestamp_string(sec):
	import datetime as dd
	d = dd.datetime.strptime("01-01-1904", "%d-%m-%Y")
	d += dd.timedelta(seconds=int(sec))
	return d.strftime("%a, %d %b %Y %H:%M:%S GMT")


########################################################################
########################################################################

ifile = h5py.File(infile, "r")
ofile = h5py.File(outfile, "w")

acq_devices = ifile["MIES"]["ITCDevices"]
dname = acq_devices["ITCPanelTitleList"].value
# pull out device name and number
# for now, require that there only be one device as we're only expecting
#   data from one device (ie, don't know what to do with a second one)
toks = dname.split(';')
assert len(toks) == 2, "Unexpected string format in '%s'" % dname
assert len(toks[1]) == 0
toks = toks[0].split('_')
dname = toks[0]

notebook = ifile["MIES"]["LabNoteBook"][dname]["Device%s" % toks[2]]
devicefolder = acq_devices[dname]["Device%s" % toks[2]]
datafolder = devicefolder["Data"]

########################################################################
# build list of sweeps to process
h_general = ofile.create_group("general")
# TODO put metadata in general

########################################################################
# build list of sweeps to process
sweep_num_list = []
for k in datafolder.keys():
	if not k.startswith("Sweep"):
		continue
	if k.endswith("DA0") or k.endswith("AD0"):
		continue
	n = int(k.split('_')[1])
	sweep_num_list.append(n)
sweep_num_list.sort()
print "Processing %d sweeps" % len(sweep_num_list)

########################################################################
# load lab notebook and summarize it
notebook_cols = notebook["KeyWave"]["keyWave"].value
notebook_vals = notebook["settingsHistory"]["settingsHistory"].value
summary = []
# find tracked columns in notebook
bridge_bal_str = "Bridge Bal Value"
bridge_bal_idx = -1
bias_current_str = "I-Clamp Holding Level"
bias_current_idx = -1
timestamp_str = "TimeStamp"
timestamp_idx = -1
sweepnum_str = "SweepNum"
sweepnum_idx = -1
for i in range(len(notebook_cols[0])):
	if notebook_cols[0][i] == bridge_bal_str:
		bridge_bal_idx = i
	if notebook_cols[0][i] == bias_current_str:
		bias_current_idx = i
	if notebook_cols[0][i] == timestamp_str:
		timestamp_idx = i
	if notebook_cols[0][i] == sweepnum_str:
		sweepnum_idx = i
assert bridge_bal_idx >= 0
assert bias_current_idx >= 0
assert timestamp_idx >= 0
assert sweepnum_idx >= 0
# read notebook backward to get final values for sweep data, ignoring
#   previous entries
for i in range(len(notebook_vals)-1, 0, -1):
	swp = int(notebook_vals[i][sweepnum_idx][0])
	while len(summary) <= swp:
		summary.append({})
	vals = summary[swp]
	vals["bridge"] = notebook_vals[i][bridge_bal_idx][0]
	vals["bias"] = notebook_vals[i][bias_current_idx][0]
	vals["time"] = notebook_vals[i][timestamp_idx][0]
	if swp == 0:
		break

########################################################################
# store high-level metadata in output file
exp_start_time = summary[0]["time"]
tstr = timestamp_string(exp_start_time)
ofile.create_dataset("experiment_start_time", data=tstr)
ofile.create_dataset("neurodata_version", data=FILE_VERSION_STR)
hstr = IDENT_PREFIX + time.ctime()
ofile.create_dataset("identifier", data=hstr)
ofile.create_dataset("file_create_date", data=time.ctime())

########################################################################
# create acquisition sequences
h_acquisition = ofile.create_group("acquisition")
h_acq_sequences = h_acquisition.create_group("sequences")
for i in range(len(sweep_num_list)):
	if TESTING and i > TEST_N:
		break	# TODO REMOVE DEBUG
	sweep = sweep_num_list[i]
	swp = "Sweep_%d" % sweep
	seq = sequence.PatchClampSequence()
	seq.set_bridge_balance(1e6 * summary[sweep]["bridge"])
	seq.set_access_resistance(1e-12 * summary[sweep]["bias"])
	# set patch-clamp sequence values
	seq.bridge_balance = summary[i]["bridge"]
	seq.bias_current = summary[i]["bias"]
	# TODO set electronic sequence values
	# set sequence values
	sweep = sweep_num_list[i]
	config = datafolder["Config_Sweep_%d" % sweep].value
	sampling_rate = config[1][2]
	# require that stimulus and voltage are sampled at the same rate
	# the code is written with this assumption 
	# if these are at different, modify the code to support this and 
	#   use the new data file to make sure nothing breaks
	assert sampling_rate == config[0][2]
	# copy sweep's trace into voltage array
	trace = datafolder[swp].value
	data = np.zeros(len(trace))
	for k in range(len(trace)):
		data[k] = 1e-3 * trace[k][1]
	seq.data = data
	seq.min_val = min(data)
	seq.max_val = max(data)
	# TODO set seq.resolution
	t0 = summary[sweep]["time"] - exp_start_time
	t1 = t0 + len(data) * sampling_rate / 1000000.0
	seq.t = []
	seq.t.append(t0)
	seq.t.append(t1)
	seq.num_samples = len(data)
	seq.t_interval = seq.num_samples - 1
	seq.write_h5(h_acq_sequences, swp)

########################################################################
# process and store stimuli

# create template stimuli first
# TODO first search lab notebook. if stimulus data not there, then 
#   try to pull it from wave notes
h_stimulus = ofile.create_group("stimulus")
h_stimulus_temps = h_stimulus.create_group("templates")
h_stimulus_present = h_stimulus.create_group("presentation")
stim = ""
scale_factor = ""
# store stimulus templates
stim_templates = {}
# list of stimulus instances, one for each sweep
stim_instance_template = []
for i in range(len(sweep_num_list)):
	if TESTING and i > TEST_N:
		break	# TODO REMOVE DEBUG
	scale_factor = None
	stim = None
	swp = "Sweep_%d" % sweep_num_list[i]
	trace = datafolder[swp].value
	attrs = datafolder[swp].attrs["IGORWaveNote"].split('\r')
	for j in range(len(attrs)):
		if attrs[j].startswith("StimWaveName"):
			stim = attrs[j].split(": ")[1]
		if attrs[j].startswith("StimScaleFactor"):
			scale_factor = float(attrs[j].split(": ")[1])
	label = "%s:%f" % (stim, scale_factor)
	if label in stim_templates:
		seq = stim_templates[label]
		seq.nwb_sweeps.append(swp)
		stim_instance_template.append(seq)
		continue
	seq = sequence.Sequence()
	seq.nwb_sweeps = []
	seq.nwb_sweeps.append(swp)
	data = np.zeros(len(trace))
	for k in range(len(trace)):
		data[k] = 1e-12 * trace[k][0]
	seq.data = data
	seq.max_val = max(data)
	seq.min_val = min(data)
	seq.num_samples = len(data)
	config = datafolder["Config_Sweep_%d" % sweep_num_list[i]].value
	seq.sampling_rate = config[1][2]
	t0 = 0
	t1 = len(data) * seq.sampling_rate / 1000000.0
	seq.t = []
	seq.t.append(t0)
	seq.t.append(t1)
	seq.t_interval = len(data)-1
	stim_instance_template.append(seq)
	stim_templates[label] = seq
	h5seq = seq.write_h5(h_stimulus_temps, label)
	seq.nwb_h5_obj = h5seq

for k in stim_templates.keys():
	seq = stim_templates[k]
	epochs = []
	for i in range(len(seq.nwb_sweeps)):
		epochs.append(np.string_(seq.nwb_sweeps[i]))
	stim_templates[k].nwb_h5_obj.attrs.create("epochs", data=epochs)


# create instance stimuli
for i in range(len(sweep_num_list)):
	if TESTING and i > TEST_N:
		break	# TODO REMOVE DEBUG
	sweep = sweep_num_list[i]
	swp = "Sweep_%d" % sweep
	seq_temp = stim_instance_template[i]
	seq_temp_h5 = seq_temp.nwb_h5_obj
	t0 = summary[sweep]["time"] - exp_start_time
	t1 = t0 + len(data) * seq_temp.sampling_rate / 1000000.0
	# create stimulus sequence using link to template's data

	seq_inst = sequence.Sequence()
	seq_inst.max_val = seq_temp.max_val
	seq_inst.min_val = seq_temp.min_val
	seq_inst.num_samples = seq_temp.num_samples
	seq_inst.sampling_rate = seq_temp.sampling_rate
	assert len(seq_temp.t) == 2
	dt = seq_temp.t[1] - seq_temp.t[0]
	seq_inst.data = seq_temp.data
	seq_inst.t = [ t0, t1 ]
	seq_inst.t_interval = seq_temp.t_interval
	seq_inst.subclass = copy.deepcopy(seq_temp.subclass)
	seq_inst.write_h5_link_data(h_stimulus_present, swp, seq_temp_h5)
	
########################################################################
# create epochs and store voltage trace
# get the sampling rate
sampling_rate = -1
# sometimes sweep times aren't reported accurately. to detect this, we
#   need to measure the reported time interval between sweeps
last_start = 0.0
last_end = 0.0
epochs = ofile.create_group("epochs")
for i in range(len(sweep_num_list)):
	if TESTING and i > TEST_N:
		break	# TODO REMOVE DEBUG
	sweep = sweep_num_list[i]
	swp = "Sweep_%d" % sweep
	config = datafolder["Config_Sweep_%d" % sweep].value
	if sampling_rate < 0:
		sampling_rate = config[1][2]
	# require that stimulus and voltage are sampled at the same rate
	# the code is written with this assumption 
	# if these are at different, modify the code to support this and 
	#   use the new data file to make sure nothing breaks
	assert sampling_rate == config[0][2]
	# create epoch and store start time
	epo = epochs.create_group(swp)
	acq = h_acq_sequences[swp]
	t = acq["timestamps"].value
	samples = len(acq["data"])
	epo.create_dataset("start_time", data=t[0])
	# add acquisition data to epoch
	acq_volts = epo.create_group("acq_voltage")
	acq_volts.create_dataset("t_start", data=t[0])
	acq_volts.create_dataset("idx_start", data=int(0))
	acq_volts.create_dataset("t_stop", data=t[1])
	acq_volts.create_dataset("idx_stop", data=samples-1)
	# advance clock for next sweep
	# TODO FIXME need actual sweep times from notebook
	#t = 1.0 + math.ceil(t)

# finish off epochs
#for i in range(len(sweep_num_list)):
#	swp = "Sweep_%d" % sweep_num_list[i]
#	# TODO for description, use stimulus name plus set number
#	# TODO link to acquisition sequence
#	# TODO add stimulus

########################################################################
# add remaining top-level groups to output file
ofile.create_group("processing")
ofile.create_group("analysis")

########################################################################
# close up shop
ofile.close()
ifile.close()

