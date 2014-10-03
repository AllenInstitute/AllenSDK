#!/usr/bin/python
import sequence
import numpy as np
import h5py

seq = sequence.Sequence()

data = np.array([ 4.5, 4.2, 4.4, 4.6, 4.9, 4.8, 4.2, 4.7, 4.6, 4.3])
t = np.array([ 1.01, 2.0, 2.97, 4.02, 6.1, 7.01, 7.95, 8.99, 10.1, 11.0])

seq.data = data
seq.set_data(data, t, 5, 1.0, 1.5)
seq.description = "test sequence"
seq.filter_desc = "minimal hardware filtering"
seq.print_report()

outf = h5py.File("bar.h5", "w")

seq.write_data(outf, "testseq")

outf.close()

