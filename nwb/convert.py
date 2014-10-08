#!/usr/bin/python
import h5py
import datetime
import time
import sys	# TODO remove debug
import numpy as np
import sequence
import math

infile = "154747.h5"
outfile = "foo.h5"

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
	if swp == 0 and len(summary[0]) > 0:
		break	# everything read -- break out
	while len(summary) <= swp:
		summary.append({})
	vals = summary[swp]
	vals["bridge"] = notebook_vals[i][bridge_bal_idx][0]
	vals["bias"] = notebook_vals[i][bias_current_idx][0]
	vals["time"] = notebook_vals[i][timestamp_idx][0]

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
acquisition = ofile.create_group("acquisition")
for i in range(len(sweep_num_list)):
	swp = "Sweep_%d" % sweep_num_list[i]
	seq = sequence.PatchClampSequence()
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
		data[k] = trace[k][1]
	seq.data = data
	seq.min_val = min(data)
	seq.max_val = max(data)
	# TODO set seq.resolution
	t0 = summary[sweep]["time"] - exp_start_time
	t1 = t0 + len(trace) * sampling_rate / 1000000.0
	seq.t = []
	seq.t.append(t0)
	seq.t.append(t1)
	seq.num_samples = len(trace)
	seq.t_interval = seq.num_samples - 1
	seq.write_data(acquisition, swp)
	if i > 5:
		break	# TODO REMOVE DEBUG

########################################################################
# process and store stimuli

sys.exit(0)

########################################################################
# create epochs and store voltage trace
#acquisition = ofile.create_group("acquisition")
# allocate storage for voltage data
cnt = 0
for i in range(len(sweep_num_list)):
	cnt += datafolder["Sweep_%d" % sweep_num_list[i]].len()
seq = sequence.Sequence()
seq.data = np.zeros(cnt)
print cnt
# fetch voltage data
pos = 0
# get the sampling rate
sampling_rate = -1
# time
t = 0.0
# sometimes sweep times aren't reported accurately. to detect this, we
#   need to measure the reported time interval between sweeps
last_start = 0.0
last_end = 0.0
epochs = ofile.create_group("epochs")
for i in range(len(sweep_num_list)):
	sweep = sweep_num_list[i]
	config = datafolder["Config_Sweep_%d" % sweep].value
	if sampling_rate < 0:
		sampling_rate = config[1][2]
	# require that stimulus and voltage are sampled at the same rate
	# the code is written with this assumption 
	# if these are at different, modify the code to support this and 
	#   use the new data file to make sure nothing breaks
	assert sampling_rate == config[0][2]
	# copy sweep's trace into voltage array
	swp = "Sweep_%d" % sweep
	trace = datafolder[swp].value
	for k in range(len(trace)):
		seq.data[pos+k] = trace[k][1]
	if i > 0:
		seq.discontinuity_idx.append(pos)
		seq.discontinuity_t.append(t)
	# create epoch and store start time
	epo = epochs.create_group(swp)
	t0 = summary[sweep][5] - exp_start_time
	if t0 <= last_start:
		t0 = last_end + 1.0
	epo.create_dataset("start_time", data=t)
	# add acquisition data to epoch
	acq = epo.create_group("acq_voltage")
	acq.create_dataset("t_start", data=t)
	acq.create_dataset("idx_start", data=pos)
	# update counters
	pos += len(trace)
	t += 1.0 * len(trace) * sampling_rate / 1000000.0
	print t
	# store epoch stop time
	epo.create_dataset("stop_time", data=t)
	acq.create_dataset("t_stop", data=t)
	acq.create_dataset("idx_stop", data=pos)
	# advance clock for next sweep
	# TODO FIXME need actual sweep times from notebook
	t = 1.0 + math.ceil(t)
	if i > 5:
		break	# TODO REMOVE DEBUG
print seq.data

acquisition = ofile.create_group("acquisition")

# finish off epochs
#for i in range(len(sweep_num_list)):
#	swp = "Sweep_%d" % sweep_num_list[i]
#	# TODO for description, use stimulus name plus set number
#	# TODO link to acquisition sequence
#	# TODO add stimulus

########################################################################
# identify and categorize stimuli
# build 2-level tree for stimuli, 1st level for type, 2nd for amplitude

# to categorize, measure polarity changes and duration of time at 
#   stimulus level. store values in list then CRC the list. this 
#   should allow amplitude-independent comparison of stimuli
# for increasing interval N, add (N) to list
# for decreasing interval N, add (-N) to list
# for flat interval N, add(N + 0.1) to list
#for i in range(len(sweep_num_list)):
#	changelist = []
#	stim = datafolder["Sweep_%d" % i].value
#stim = datafolder["Sweep_10"][0]
#stim = datafolder["Sweep_10"]
#print stim.len()


########################################################################
# close up shop
ofile.close()
ifile.close()

