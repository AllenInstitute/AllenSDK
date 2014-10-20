#!/usr/bin/python
import h5py
import datetime
import copy
import time
import sys
import os
import glob
import numpy as np
import sequence
import math
from aif_common import *

# TODO remove testing code
TESTING = True
TEST_N = 199

VERS_MAJOR = 0
VERS_MINOR = 9
VERS_PATCH = 0

FILE_VERSION_STR = "AIFile.%d.%d.%d" % (VERS_MAJOR, VERS_MINOR, VERS_PATCH)

IDENT_PREFIX = "Allen Institute " + FILE_VERSION_STR + ": "

PULSE_LEN = 0.5

def convert_h5d_aif(infilei, outfile):
	print "Generating " + outfile

	########################################################################
	########################################################################
	# analysis prep

	# make list of image files to add to acquisition
	rootdir = os.path.dirname(os.path.realpath(infile))
	rootfile = None
	imagefiles = {}
	if infile.endswith(".h5"):
		rootfile = infile[:-3]
		lst = glob.glob(rootfile + "*.tif")
		for i in range(len(lst)):
			imagefiles[os.path.basename(lst[i])] = lst[i]


	########################################################################
	########################################################################
	# create input and output files
	ifile = h5py.File(infile, "r")
	ofile = h5py.File(outfile, "w")


	# TODO verify input file is supported version


	# find where data is located in file
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

	# make reference to notebook and data folders
	notebook = ifile["MIES"]["LabNoteBook"][dname]["Device%s" % toks[2]]
	devicefolder = acq_devices[dname]["Device%s" % toks[2]]
	datafolder = devicefolder["Data"]


	########################################################################
	# TODO create and populate general folder
	h_general = ofile.create_group("general")


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
	n = len(sweep_num_list)
	if TESTING:
		if TEST_N < n:
			n = TEST_N
	print "Processing %d sweeps" % n


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
	scalefactor_str = "Stim Scale Factor"
	scalefactor_idx = -1
	for i in range(len(notebook_cols[0])):
		if notebook_cols[0][i] == bridge_bal_str:
			bridge_bal_idx = i
		if notebook_cols[0][i] == bias_current_str:
			bias_current_idx = i
		if notebook_cols[0][i] == timestamp_str:
			timestamp_idx = i
		if notebook_cols[0][i] == sweepnum_str:
			sweepnum_idx = i
		if notebook_cols[0][i] == scalefactor_str:
			scalefactor_idx = i
	assert bridge_bal_idx >= 0
	assert bias_current_idx >= 0
	assert timestamp_idx >= 0
	assert sweepnum_idx >= 0
	assert scalefactor_idx >= 0
	# read notebook backward to get final values for sweep data, ignoring
	#   previous entries
	# last value is what matters, so read from front and overwrite any
	#   previous value for a given sweep
	for i in range(len(notebook_vals)):
		#print notebook_vals[i][sweepnum_idx]
		swp_num = notebook_vals[i][sweepnum_idx][0]
		if math.isnan(swp_num):
			continue
		swp_num = int(swp_num)
		while len(summary) <= swp_num:
			summary.append({})
		vals = summary[swp_num]
		bridge = notebook_vals[i][bridge_bal_idx][0]
		if not math.isnan(bridge):
			vals["bridge"] = notebook_vals[i][bridge_bal_idx][0]
		bias = notebook_vals[i][bias_current_idx][0]
		if not math.isnan(bias):
			vals["bias"] = bias
		t = notebook_vals[i][timestamp_idx][0]
		if not math.isnan(t):
			vals["time"] = t

	# summarize stimuli
	# pull stim names and associated scale factors
	sweep_stimuli = []
	# format of array
	#   one entry for each sweep, storing a dict
	#   each dict contains:
	#     'stim'
	#     'scalefactor'
	#
	# to pull notebook values it's necessary to find which column data is 
	#   stored in. search names to get indices
	sweepnum_str = "Sweep #"
	sweepnum_idx = -1
	stim_name_str = "Stim Wave Name"
	stim_name_idx = -1
	# stim_notebook_cols shows the order of columns displayed in stim_notebook
	stim_notebook_cols = notebook["TextDocKeyWave"]["txtDocKeyWave"].value
	stim_notebook = notebook["textDocumentation"]["txtDocWave"].value
	for i in range(len(stim_notebook_cols[0])):
		val = stim_notebook_cols[0][i]
		if val == stim_name_str:
			stim_name_idx = i
		if val == sweepnum_str:
			sweepnum_idx = i
	# make sure necessary cols found
	assert stim_name_idx >= 0
	assert sweepnum_idx >= 0
	# get stim name from stim notebook
	for i in range(len(stim_notebook)):
		swp_num = stim_notebook[i][sweepnum_idx][0]
		if len(swp_num) == 0:
			continue
		swp_num = int(swp_num)
		if len(sweep_stimuli) <= swp_num:
			sweep_stimuli.append({})
		sweep_stimuli[swp_num]["stim"] = stim_notebook[i][stim_name_idx][0]
	# get scale factor from regular notebook
	for i in range(len(notebook_vals)):
		swp_num = notebook_vals[i][sweepnum_idx][0]
		if math.isnan(swp_num):
			continue
		swp_num = int(swp_num)
		scale = notebook_vals[i][scalefactor_idx][0]
		if math.isnan(scale):
			continue
		sweep_stimuli[swp_num]["scalefactor"] = scale


	########################################################################
	# store high-level metadata in output file
	exp_start_time = summary[0]["time"]
	tstr = timestamp_string(exp_start_time)
	ofile.create_dataset("experiment_start_time", data=tstr)
	ofile.create_dataset("ai_version", data=FILE_VERSION_STR)
	hstr = IDENT_PREFIX + time.ctime()
	ofile.create_dataset("identifier", data=hstr)
	ofile.create_dataset("file_create_date", data=time.ctime())

	########################################################################
	# create acquisition sequences
	h_acquisition = ofile.create_group("acquisition")

	# copy experiment images to file
	h_acq_images = h_acquisition.create_group("images")
	for k in imagefiles.keys():
		with open(imagefiles[k], mode='rb') as f:
			img = np.string_(f.read())
			h_acq_images.create_dataset(k, data=img)
			f.close()

	# copy acquired data
	h_acq_sequences = h_acquisition.create_group("sequences")
	for i in range(len(sweep_num_list)):
		if TESTING and i > TEST_N:
			break	# TODO REMOVE DEBUG
		swp_num = sweep_num_list[i]
		swp_name = "Sweep_%d" % swp_num
		seq = sequence.PatchClampSequence()
		stim_name = sweep_stimuli[swp_num]["stim"]
		stim_mag = sweep_stimuli[swp_num]["scalefactor"]
		###############################
		# TODO Remove the following code when notebook's stim scale factor works
		# meansure peak stimulus
		# find stimulus start. start looking 0.30 seconds into stim
		stim_mag = 0.0
		trace = datafolder[swp_name].value
		for j in range(int(0.3/5e-6), len(trace)):
			val = trace[j][0]
			if abs(val) > abs(stim_mag):
				stim_mag = val
		sweep_stimuli[swp_num]["scalefactor"] = "%3g" % stim_mag
		###############################
		# use stimulus name and scale factor in sequence description
		seq.description = "trace for %s:%s" % (stim_name, stim_mag)
		seq.set_bridge_balance(1e6 * summary[swp_num]["bridge"])
		seq.set_access_resistance(1e-12 * summary[swp_num]["bias"])
		# set patch-clamp sequence values
		seq.bridge_balance = summary[i]["bridge"]
		seq.bias_current = summary[i]["bias"]
		# set sequence values
		swp_num = sweep_num_list[i]
		config = datafolder["Config_Sweep_%d" % swp_num].value
		sampling_rate = config[1][2]
		# require that stimulus and voltage are sampled at the same rate
		# the code is written with this assumption 
		# if these are at different, modify the code to support this and 
		#   use the new data file to make sure nothing breaks
		assert sampling_rate == config[0][2]
		sampling_rate /= 1.0e6
		# copy sweep's trace into voltage array
		trace = datafolder[swp_name].value
		data = np.zeros(len(trace))
		for k in range(len(trace)):
			data[k] = 1e-3 * trace[k][1]
		seq.data = data
		seq.min_val = min(data)
		seq.max_val = max(data)
		# TODO set seq.resolution
		t0 = summary[swp_num]["time"] - exp_start_time
		t1 = t0 + len(data) * sampling_rate
		seq.t = []
		seq.t.append(t0)
		seq.t.append(t1)
		seq.sampling_rate = sampling_rate
		seq.num_samples = len(data)
		seq.t_interval = seq.num_samples - 1
		seq.write_h5(h_acq_sequences, swp_name)

	########################################################################
	# process and store stimuli

	# create template stimuli first
	# TODO first search lab notebook. if stimulus data not there, then 
	#   try to pull it from wave notes
	h_stimulus = ofile.create_group("stimulus")
	h_stimulus_temps = h_stimulus.create_group("templates")
	h_stimulus_present = h_stimulus.create_group("presentation")
	# store stimulus templates
	stim_templates = {}
	# list of stimulus instances, one for each sweep
	stim_instance_template = []
	for i in range(len(sweep_num_list)):
		if TESTING and i > TEST_N:
			break	# TODO REMOVE DEBUG
		swp_num = sweep_num_list[i]
		swp_name = "Sweep_%d" % swp_num
		stim = sweep_stimuli[swp_num]["stim"]
		scale_factor = sweep_stimuli[swp_num]["scalefactor"]
		label = "%s:%s" % (stim, scale_factor)
		if label in stim_templates:
			# template already defined -- add this sweep instance to it
			seq = stim_templates[label]
			seq.nwb_sweeps.append(np.string_(swp_name))
			stim_instance_template.append(seq)
			continue
		seq = sequence.Sequence()
		seq.description = np.string_(label)
		# use custom data field in sequence, for admin purposes
		seq.nwb_sweeps = []
		seq.nwb_sweeps.append(swp_name)
		# make local copy of data, converting to SI units
		trace = datafolder[swp_name].value
		data = np.zeros(len(trace))
		for k in range(len(trace)):
			data[k] = 1e-12 * trace[k][0]
		seq.data = data
		seq.max_val = max(data)
		seq.min_val = min(data)
		seq.num_samples = len(data)
		config = datafolder["Config_Sweep_%d" % sweep_num_list[i]].value
		seq.sampling_rate = 1.0e-6 * config[1][2]
		t0 = 0
		t1 = len(data) * seq.sampling_rate
		seq.t = []
		seq.t.append(t0)
		seq.t.append(t1)
		seq.t_interval = len(data)-1
		stim_instance_template.append(seq)
		stim_templates[label] = seq
		h5seq = seq.write_h5(h_stimulus_temps, label)
		seq.nwb_h5_obj = h5seq

	data = None

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
		swp_num = sweep_num_list[i]
		swp_name = "Sweep_%d" % swp_num
		seq_temp = stim_instance_template[i]
		data = seq_temp.data
		seq_temp_h5 = seq_temp.nwb_h5_obj
		t0 = summary[swp_num]["time"] - exp_start_time
		t1 = t0 + len(data) * seq_temp.sampling_rate
		# create stimulus sequence using link to template's data
		seq_inst = sequence.Sequence()
		seq_inst.max_val = seq_temp.max_val
		seq_inst.min_val = seq_temp.min_val
		seq_inst.num_samples = seq_temp.num_samples
		seq_inst.sampling_rate = seq_temp.sampling_rate
		seq_inst.description = seq_temp.description
		assert len(seq_temp.t) == 2
		dt = seq_temp.t[1] - seq_temp.t[0]
		seq_inst.data = seq_temp.data
		seq_inst.t = [ t0, t1 ]
		seq_inst.t_interval = seq_temp.t_interval
		seq_inst.subclass = copy.deepcopy(seq_temp.subclass)
		seq_inst.write_h5_link_data(h_stimulus_present, swp_name, seq_temp_h5)
		
	########################################################################
	# create epochs and store voltage trace

	# helper function -- inserts sequence reference to epoch
	def epoch_insert_sequence(epo, name, t0, t1, idx0, idx1, desc, seq):
		# add acquisition data to epoch
		epo_seq = epo.create_group(name)
		epo_seq.create_dataset("t_start", data=t0)
		epo_seq.create_dataset("idx_start", data=idx0)
		epo_seq.create_dataset("t_stop", data=t1)
		epo_seq.create_dataset("idx_stop", data=idx1)
		epo_seq["sequence"] = seq
		
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
		swp_num = sweep_num_list[i]
		swp_name = "Sweep_%d" % swp_num
		config = datafolder["Config_Sweep_%d" % swp_num].value
		if sampling_rate < 0:
			sampling_rate = config[1][2]
		# require that stimulus and voltage are sampled at the same rate
		# the code is written with this assumption 
		# if these are at different, modify the code to support this and 
		#   use the new data file to make sure nothing breaks
		assert sampling_rate == config[0][2]
		# create epoch and store start time
		epo = epochs.create_group(swp_name)
		acq = h_acq_sequences[swp_name]
		t = acq["timestamps"].value
		samples = len(acq["data"])
		t0 = t[0]
		t1 = t[1]
		idx0 = 0
		idx1 = samples - 1
		stim = sweep_stimuli[swp_num]["stim"]
		desc = sweep_stimuli[swp_num]["scalefactor"]
		epo.create_dataset("start_time", data=t0)
		epo.create_dataset("stop_time", data=t1)
		epo.create_dataset("description", data=np.string_(desc))
		intervals = [[0][0]]
		epo.create_dataset("ignore_intervals", data=intervals)
		# add acquisition data to epoch
		seq = h_acq_sequences[swp_name]
		desc = "%s test-pulse voltage" % swp_name
		name = "acq_voltage"
		epoch_insert_sequence(epo, name, t0, t1, idx0, idx1, desc, seq)
		# add stim sequence
		seq = h_stimulus["presentation"][swp_name]
		desc = "%s test-pulse current" % swp_name
		name = "stim_current"
		epoch_insert_sequence(epo, name, t0, t1, idx0, idx1, desc, seq)


	# create sub-epochs, one showing test pulse and the other showing stim
	for i in range(len(sweep_num_list)):
		if TESTING and i > TEST_N:
			break	# TODO REMOVE DEBUG
		swp_num = sweep_num_list[i]
		swp_name = "Sweep_%d" % swp_num
		stim_name = "Stim_%d" % swp_num
		config = datafolder["Config_Sweep_%d" % swp_num].value
		orig_v = epochs[swp_name]["acq_voltage"]
		orig_i = epochs[swp_name]["stim_current"]
		sampling_rate = orig_v["sequence"]["sampling_rate"].value
		epo = epochs[swp_name]
		# don't write sub-epochs for smoke tests, or stims that are too
		#   short 
		idx0 = int(1.0 / sampling_rate)
		idx1 = len(h_acq_sequences[swp_name]["data"])
		if int(1.0 / sampling_rate) > len(h_acq_sequences[swp_name]["data"]):
			continue
		##############################
		# pulse epoch
		#
		# pulse duration is 100ms
		pulse_name = "Pulse_%d" % swp_num
		t0 = orig_v["t_start"].value
		t1 = t0 + PULSE_LEN 
		idx0 = 0
		idx1 = int(PULSE_LEN / sampling_rate)
		epo_pulse = epochs.create_group(pulse_name)
		epo_pulse.create_dataset("start_time", data=t0)
		epo_pulse.create_dataset("stop_time", data=t1)
		epo_pulse.create_dataset("start_idx", data=idx0)
		epo_pulse.create_dataset("stop_idx", data=idx1)
		desc = "Test pulse for " + swp_name
		epo_pulse.create_dataset("description", data=np.string_(desc))
		intervals = [[0][0]]
		epo_pulse.create_dataset("ignore_intervals", data=intervals)
		# add acq sequence
		seq = h_acq_sequences[swp_name]
		desc = "%s test-pulse voltage" % swp_name
		name = "acq_voltage"
		epoch_insert_sequence(epo_pulse, name, t0, t1, idx0, idx1, desc, seq)
		# add stim sequence
		seq = h_stimulus["presentation"][swp_name]
		desc = "%s test-pulse current" % swp_name
		name = "stim_current"
		epoch_insert_sequence(epo_pulse, name, t0, t1, idx0, idx1, desc, seq)
		##############################
		# stim epoch
		#
		# pulse duration is 100ms
		pulse_name = "Stim_%d" % swp_num
		t0 = orig_v["t_start"].value + 1.0
		t1 = orig_v["t_stop"].value
		idx0 = int(1.0 / sampling_rate)
		idx1 = len(h_acq_sequences[swp_name]["data"])
		epo_pulse = epochs.create_group(pulse_name)
		epo_pulse.create_dataset("start_time", data=t0)
		epo_pulse.create_dataset("stop_time", data=t1)
		epo_pulse.create_dataset("start_idx", data=idx0)
		epo_pulse.create_dataset("stop_idx", data=idx1)
		desc = "Experiment data for " + swp_name
		epo_pulse.create_dataset("description", data=np.string_(desc))
		intervals = [[0][0]]
		epo_pulse.create_dataset("ignore_intervals", data=intervals)
		# add acq sequence
		seq = h_acq_sequences[swp_name]
		desc = "%s test-pulse voltage" % swp_name
		name = "acq_voltage"
		epoch_insert_sequence(epo_pulse, name, t0, t1, idx0, idx1, desc, seq)
		# add stim sequence
		seq = h_stimulus["presentation"][swp_name]
		desc = "%s test-pulse current" % swp_name
		name = "stim_current"
		epoch_insert_sequence(epo_pulse, name, t0, t1, idx0, idx1, desc, seq)


	########################################################################
	# add remaining top-level groups to output file
	ofile.create_group("processing")
	ofile.create_group("analysis")

	########################################################################
	# close up shop
	ofile.close()
	ifile.close()


if __name__ == "__main__":
	if len(sys.argv) != 2:
		if TESTING:
			infile = "157436/157436.03.01.h5"
		else:
			print "Usage: %s <input h5>" % sys.argv[0]
			sys.exit(1)
	else:
		TESTING = False
		infile = sys.argv[1]

	dirlen = len(os.path.dirname(infile))
	if dirlen > 1:
		outfile = infile[dirlen+1:]
	else:
		outfile = infile
	if outfile.endswith(".h5"):
		outfile = outfile[:-3]
	outfile = outfile + ".ai"

	convert_h5d_aif(infile, outfile)

