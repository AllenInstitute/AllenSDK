import sys
import copy
import numpy as np
import traceback

class Epoch(object):
    """ Epoch object
        Epochs represent specific experimental intervals and store
        references to desired time series that overlap with the interval.
        The references to those time series indicate the first
        index in the time series that overlaps with the interval, and the
        duration of that overlap.

        **Constructor argumetns:**
            **name** Name of epoch (must be unique among epochs)
            
            **nwb** NWB object (class)

            **start** Start time of epoch, in seconds

            **stop** Stop time of epoch, in seconds
    """
    def __init__(self, name, nwb, start, stop, spec):
        # dict to keep track of which time series are linked to this epoch
        self.timeseries_dict = {}
        # start and stop time (in seconds)
        self.start_time = start
        self.stop_time = stop
        # name of epoch
        self.name = name
        # make a copy of the epoch specification
        self.spec = copy.deepcopy(spec)
        # reference to nwb 'kernel'
        self.nwb = nwb
        # intervals that are for ignoring data (eg, due noise)
        self.spec["ignore_intervals"]["_value"] = []
        # list of tags associated with epoch
        self.spec["tags"]["_value"] = []
        self.spec["_attributes"]["links"]["_value"] = []
        # go ahead and create epoch folder now
        epoch = nwb.file_pointer["epochs"].create_group(self.name)
        self.finalized = False

    def set_description(self, desc):
        self.set_value("description", value)

    def set_value(self, key, value, **attrs):
        """Adds an annotating key-value pair (ie, dataset) to the epoch.
   
           Args:
               *key* (string) A unique identifier within the TimeSeries

               *value* (any) The value associated with this key
   
           Returns:
               *nothing*
        """
        if self.finalized:
            nwb.fatal_error("Added value to epoch after finalization")
        self.spec[key] = copy.deepcopy(self.spec["[]"])
        dtype = self.spec[key]["_datatype"]
        name = "epoch " + self.name
        self.nwb.set_value_internal(key, value, self.spec, name, dtype, **attrs)

    def add_tag(self, content):
        self.spec["tags"]["_value"].append(content)

    # limit intervals to time boundary of epoch, but don't perform 
    #   additional logic (ie, if user supplies overlapping intervals,
    #   let them do so
    def add_ignore_interval(self, start, stop):
        if start > self.stop_time or stop < self.start_time:
            return  # non-overlapping
        if start < self.start_time:
            start = self.start_time
        if stop > self.stop_time:
            stop = self.stop_time
        self.spec["ignore_intervals"]["_value"].append([start, stop])

    def add_timeseries(self, in_epoch_name, timeseries_path):
        """ Associates time series with epoch. This will create a link
            to the specified time series within the epoch and will
            calculate its overlaps.

            Args:
                *in_epoch_name* (text) Name that time series will use 
                in the epoch (this can be different than the actual 
                time series name)

                *timeseries_path* (text) Full hdf5 path to time series
                that's being added

            Returns:
                *nothing*
        """
        # store path to timeseries, so can create hard link
        epoch_ts = {}
        epoch_ts["timeseries"] = timeseries_path
        #print timeseries_path
        if timeseries_path not in self.nwb.file_pointer:
            self.nwb.fatal_error("Time series '%s' not found" % timeseries_path)
        ts = self.nwb.file_pointer[timeseries_path]
        if "timestamps" in ts:
            t = ts["timestamps"].value
        else:
            n = ts["num_samples"].value
            t0 = ts["starting_time"].value
            rate = ts["starting_time"].attrs["rate"]
            t = t0 + np.arange(n) / rate
        # if no overlap, don't add to timeseries
        # look for overlap between epoch and time series
        i0, i1 = self.find_ts_overlap(t)
        if i0 is None:
            return
        epoch_ts["start_idx"] = i0
        epoch_ts["count"] = i1 - i0 + 1
        self.timeseries_dict[in_epoch_name] = epoch_ts
        label = in_epoch_name + " is " + timeseries_path
        self.spec["_attributes"]["links"]["_value"].append(label)

    def set_description(self, desc):
        """ This sets the epoch's description field (a required field
            but that can be empty)

            Args:
                *desc* (text) Textual description 

            Returns:
                *nothing*
        """
        self.spec["description"]["_value"] = desc
        
    def find_ts_overlap(self, timestamps):
        """ Finds the first element in *timestamps* that is >= *epoch_start*
            and last element that is <= "epoch_stop"

            Args:
                *timestamps* (double array) Timestamp array

            Returns:
                *idx_0*, "idx_1" (ints) Index of first and last elements 
                in *timestamps* that fall within specified
                interval, or None, None if there is no overlap
        """
        start = self.start_time
        stop = self.stop_time
        # ensure there are non-nan times
        isnan = np.isnan(timestamps)
        if isnan.all(): 
            return None, None   # all values are NaN -- no overlap
        # convert all nans to a numerical value 
        # when searching for start, use -1
        # when searching for end, use stop+1
        timestamps = np.nan_to_num(timestamps)
        t_test = timestamps + isnan * -1 # make nan<0
        # array now nan-friendly. find first index where timestamps >= start
        i0 = np.argmax(t_test >= start)
        # if argmax returns zero then the first timestamp is >= start or 
        #   no timestamps are. find out which is which
        if i0 == 0:
            if t_test[0] < start:
                return None, None # no timestamps > start
        if t_test[i0] > stop:
            return None, None # timestamps only before start and after stop
        # if we reached here then some or all timestamps are after start
        # search for first after end, adjusting compare array so nan>stop
        t_test = timestamps + isnan * (stop+1)
        # start looking after i0 -- no point looking before, plus if we
        #   do look before and NaNs are present, it screws up the search
        i1 = np.argmin((t_test <= stop)[i0:])
        # if i1 is 0, either all timestamps are <= stop, or all > stop
        if i1 == 0:
            if t_test[0] > stop:
                return None, None # no timestamps < stop
            i1 = len(timestamps) - 1
        else:
            i1 = i0 + i1 - 1 # i1 is the first value > stop. fix that
        # make sure adjusted i1 value is non-nan
        while isnan[i1]:
            i1 = i1 - 1
            assert i1 >= 0
        try:    # error checking 
            assert i0 <= i1
            assert not np.isnan(timestamps[i0])
            assert not np.isnan(timestamps[i1])
            assert timestamps[i0] >= start and timestamps[i0] <= stop
            assert timestamps[i1] >= start and timestamps[i1] <= stop
            return i0, i1
        except AssertionError:
            print "-------------------" + self.name
            print "epoch: %f, %f" % (start, stop)
            print "time: %f, %f" % (timestamps[0], timestamps[-1])
            print "idx 0: %d\tt:%f" % (i0, timestamps[i0])
            print "idx 1: %d\tt:%f" % (i1, timestamps[i1])
            assert False, "Internal error"

    def finalize(self):
        """ Finish epoch entry and write data to the file

            Args:
                *none*

            Returns:
                *nothing*
        """
        if self.finalized:
            return
        # start and stop time stored outside of spec 
        # put them in for writing
        self.spec["start_time"]["_value"] = self.start_time
        self.spec["stop_time"]["_value"] = self.stop_time
        # write epoch entry
        fp = self.nwb.file_pointer
        epoch = fp["epochs"][self.name]
        # manually create time series references
        for k in self.timeseries_dict.keys():
            ts = self.timeseries_dict[k]
            ets = epoch.create_group(k)
            src = self.timeseries_dict[k]["timeseries"]
            ets["timeseries"] = fp[src]
            ets.create_dataset("idx_start", data=ts["start_idx"], dtype='i4')
            ets.create_dataset("count", data=ts["count"], dtype='i4')
        # report all tags to kernel so it can keep track of what was used
        self.nwb.add_epoch_tags(self.spec["tags"]["_value"])
        # write content to file
        grp = self.nwb.file_pointer["epochs/" + self.name]
        self.nwb.write_datasets(grp, "", self.spec)
        self.nwb.write_attributes(grp, self.spec)
        # flag ourself as done
        self.finalized = True

