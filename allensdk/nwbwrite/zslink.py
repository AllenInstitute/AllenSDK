#!/usr/bin/python
import sys
import nwb

# create a new borg file. If we don't specify a time, the present time
#   will be used
borg = nwb.NWB(filename="sample_interval_series2.nwb", identifier="test", overwrite=True, description="Test file for soft links")

# create an IntervalSeries
# This will normally be stored in processing or stimulus
# For this example, store it in stimulus (keeps things more simple)
interval = borg.create_timeseries("IntervalSeries", "intervals", "stimulus")
interval.set_description("Sample interval series -- two series are overlaid here, one with a code '1' and another with the code '2'")
interval.set_comment("For example, '1' represents sound on/off and '2' represents light on/off")
# for data, link to output of zannot.py
interval.set_data_as_remote_link("sample_interval_series.nwb", "stimulus/presentation/intervals/data")
t = [ 1, 2, 2, 3, 5, 6, 6, 7, 8, 8, 10, 10, 11, 15 ]
interval.set_time(t)
# 
interval.finalize()

# when all data is entered, close the Borg file
borg.close()

