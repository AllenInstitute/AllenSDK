import numpy as np
import h5py
import math

class Sequence():
	def __init__(self):
		self.data = None
		self.description = ""
		self.sampling_rate = 0
		self.num_samples = 0
		self.resolution = None
		self.filter_desc = ""
		self.t = None
		self.max_val = 0
		self.min_val = 0
		self.t_interval = 1
		self.discontinuity_t = []
		self.discontinuity_idx = []

	def print_report(self):
		print "Description:   '%s'" % self.description
		print "Sampling rate: %d" % self.sampling_rate
		print "Num samples:   %d" % self.num_samples
		if self.resolution == None:
			self.resolution = 1e-20
		print "Resolution:    %g" % self.resolution
		print "Filter:        '%s'" % self.filter_desc
		print "Max value:     %g" % self.max_val
		print "Min value:     %g" % self.min_val
		print "Interval:      %d" % self.t_interval
		print "Data:"
		ctr = 0
		for i in range(len(self.t)):
			print "\t%g" % self.t[i]
			for j in range(self.t_interval):
				print "\t\t%g" % self.data[ctr]
				ctr += 1
		print "Discontinuities:"
		for i in range(len(self.discontinuity_t)):
			print "%8d\t%g" % (self.discontinuity_idx[i], self.discontinuity_t[i])
		
	def set_data(self, data, t, interval, dt, dt_err):
		# t is array of all times
		# interval is space between successive timestamps
		# dt is expected interval between samples
		# dt_err is max error before flagging discontinuity
		# 
		# sets t, t_interval, discontinuity_t, discontinuity_idx
		#
		assert len(data) == len(t)
		self.data = data
		# create timestamp array
		self.t = np.zeros(math.ceil(len(t)/interval))
		for i in range(len(self.t)):
			self.t[i] = t[i*interval]
		self.t_interval = interval
		self.sampling_rate = 1.0 / dt
		# identify discontinuities
		for i in range(1, len(t)):
			if t[i] - t[i-1] > dt_err:
				self.discontinuity_t.append(t[i])
				self.discontinuity_idx.append(i)
		# store number of samples
		self.num_samples = len(t)
		# calculate peaks
		self.max_val = max(data)
		self.min_val = min(data)


	#def write_data(parent, seq_name):
	def write_data(self, parent, seq_name):
		assert seq_name not in parent
		seq = parent.create_group(seq_name)
		meta = seq.create_group("meta")
		strlen = len(self.description)
		seq.create_dataset("description", data=self.description)
		seq.create_dataset("resolution", data=self.resolution)
		seq.create_dataset("data", data=self.data, dtype='f4')
		seq.create_dataset("t", data=self.t, dtype='f8')
		dis_t = self.discontinuity_t
		seq.create_dataset("discontinuity_t", data=dis_t, dtype='f8')
		dis_i = self.discontinuity_idx
		seq.create_dataset("discontinuity_idx", data=dis_i)
		seq.create_dataset("filter", data=self.filter_desc)
		seq.create_dataset("num_samples", data=(len(self.data)))
		seq.create_dataset("max_value", data=(max(self.data)))
		seq.create_dataset("min_value", data=(min(self.data)))
		seq.create_dataset("sampling_rate", data=self.sampling_rate)
		return seq, meta

	def read(parent, name):
		None
		

class ElectronicSequence(Sequence):
	def __init__():
		super.__init__()
		self.electrode_map = None

	def write_data(parent, seq_name):
		seq, meta = super.write_data(parent, seq_name)
		meta.create_dataset("electrode_map", self.electrode_map)
		return seq, meta

class PatchClampSequence(ElectronicSequence):
	def __init__():
		super.__init__()
		self.bridge_balance = 0
		self.bias_current = 0

	def write_data(parent, seq_name):
		seq, meta = super.write_data(parent, seq_name)
		meta.create_dataset("bridge_balance", self.bridge_balance)
		meta.create_dataset("bias_current", self.bias_current)
		return seq, meta

