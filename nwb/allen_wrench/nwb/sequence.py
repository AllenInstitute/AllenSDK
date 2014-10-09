import numpy as np
import h5py
import math

class Sequence(object):
	"""
	The Sequence object represents the common storage object for the 
	nwb file. It can represent time series, signal events, image stacks
	and experimental events.
	"""
	def __init__(self):
		self.data = None
		self.image_data = None
		self.max_val = 0
		self.min_val = 0
		self.num_samples = 0
		self.resolution = 1e-20
		#
		self.description = ""
		self.filter_desc = ""
		#
		self.t = None
		self.sampling_rate = 0
		self.t_interval = 1
		self.discontinuity_t = []
		self.discontinuity_idx = []
		self.subclass = {}

	def print_report(self):
		print "Description:   '%s'" % self.description
		print "Sampling rate: %d" % self.sampling_rate
		print "Num samples:   %d" % self.num_samples
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
		"""
		data is array of all data elements
		t is array of all times
		interval is space between successive timestamps
		dt is expected interval between samples
		dt_err is max error before flagging discontinuity
		
		sets object values, except description fields and data resolution
		"""
		# sanity check
		assert len(data) == len(t)
		#############################
		# store data
		self.data = data
		self.max_val = max(data)
		self.min_val = min(data)
		self.num_samples = len(t)
		#############################
		# store temporal info
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

	def write_h5_link_data(self, parent, seq_name, sibling):
		""" create a new dataset, but link its data entry to
		the data field in its sibling
		"""
		assert seq_name not in parent
		seq = self.write_h5_no_data(parent, seq_name)
		seq["data"] = sibling["data"]
		return seq

	def write_h5_no_data(self, parent, seq_name):
		seq = parent.create_group(seq_name)
		subclass = seq.create_group("subclass")
		strlen = len(self.description)
		seq.create_dataset("description", data=self.description)
		seq.create_dataset("resolution", data=self.resolution)
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
		for k in self.subclass.keys():
			seq.create_dataset("subclass/" + k, data=self.subclass[k])
		return seq

	def write_h5(self, parent, seq_name):
		""" create new dataset for sequence and write data 
		"""
		assert seq_name not in parent
		seq = self.write_h5_no_data(parent, seq_name)
		seq.create_dataset("data", data=self.data, dtype='f4')
		return seq

		

class ElectronicSequence(Sequence):
	def __init__(self):
		super(ElectronicSequence, self).__init__()
		electrode_map = []
		electrode_map.append(0)
		self.subclass["electrode_map"] = electrode_map

#	def write_h5(self, parent, seq_name):
#		sup = super(ElectronicSequence,self)
#		seq, subclass = sup.write_h5(parent, seq_name)
#		subclass.create_dataset("electrode_map", self.electrode_map)
#		return seq, subclass

class PatchClampSequence(ElectronicSequence):
	def __init__(self):
		super(PatchClampSequence, self).__init__()
		self.subclass["bridge_balance"] = 0
		self.subclass["access_resistance"] = 0

	def set_bridge_balance(self, val):
		self.subclass["bridge_balance"] = val

	def set_access_resistance(self, val):
		self.subclass["access_resistance"] = val

#	def write_h5(self, parent, seq_name):
#		sup = super(PatchClampSequence, self)
#		seq, subclass = sup.write_h5(parent, seq_name)
#		subclass.create_dataset("bridge_balance", data=self.bridge_balance)
#		subclass.create_dataset("bias_current", data=self.bias_current)
#		return seq, subclass

