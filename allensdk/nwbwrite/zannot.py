#!/usr/bin/python
import sys
import nwb
from nwbco import *

# create a new borg file. If we don't specify a time, the present time
#   will be used
fname = "sample_zannot.nwb"
borg = nwb.NWB(filename=fname, identifier="test",overwrite=True, description="Sample annotation test script")
print "Creating " + fname
borg.set_metadata_from_file("source_script", sys.argv[0])

# create an AnnotationSeries
# This will be sotred in 'acquisiiton' as annotations are an
#   observation or a record of something else that happened 
#   (i.e., the Annotation didn't change the experimental environment)
annot = borg.create_timeseries("AnnotationSeries", "notes", "acquisition")
# create dummy entries at Fibonacci times
prev1 = -1.0
prev2 = -1.0
for i in range(10):
    t = 1.0
    if prev1 < 0:
        prev1 = 1.0
    elif prev2 < 0:
        prev2 = 1.0
    else:
        t = prev1 + prev2
        prev2 = prev1
        prev1 = t
    annot.add_annotation("dummy entry %d" % i, t)

# add a description
annot.set_description("This is an AnnotationSeries with sample data")
annot.set_comment("The comment and description fields can store arbitrary human-readable data")
annot.set_source("Source of data is the sample file " + __file__)
# finalize the time series
annot.finalize()

# when all data is entered, close the Borg file
borg.close()

