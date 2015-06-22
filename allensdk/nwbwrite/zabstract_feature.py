#!/usr/bin/python
import sys
import numpy as np
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_abstract_feature_series.nwb", identifier="test", overwrite=True, description="test abstract feature file")

# create an AbstractFeatureSeries
# This will be stored in 'stimulus' as its normal use case will be to
#   store abstract features of a stimulus (note: it can be used during
#   processing as well)
abstract = borg.create_timeseries("AbstractFeatureSeries", "abstract_features", "stimulus")
features = [ "inner-orientation", "inner-spatial frequency", "inner-temporal frequency", "outer-orientation", "outer-spatial frequency", "outer-temporal frequency" ]
units = [ "degrees", "Hz", "Hz", "degrees", "Hz", "Hz" ]
abstract.set_value("features", features)
abstract.set_value("feature_units", units)
abstract.set_description("This is a simulated visual stimulus that presents one moving grating inside another")
abstract.set_source("Data source would be device presenting stimulus, in this example")
data = np.arange(6000).reshape(1000, 6)
abstract.set_data(data, "n/a", 1, 1)
t = np.arange(1000) * 0.001
abstract.set_time(t)
# finalize the time series
abstract.finalize()

# when all data is entered, close the Borg file
borg.close()

