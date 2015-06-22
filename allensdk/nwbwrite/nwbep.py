import sys
import copy
import numpy as np
import traceback

def ts_self_check():
    """Debug code -- tests interval overlap detection/calculation
    """
    t = [ 11, 12, 13, 14, 15, 19, 20, 21, 22, 24 ]
    #self.epoch = Epoch("self test epoch", None, 0, 0)
    # format: (t, epoch_start, epoch_end, idx0, cnt)
    # TS fully within epoch
    test_set(t, 0, 30, 0, 10)
    # TS start before epoch, end during epoch on TS point
    test_set(t, 15, 30, 4, 6)
    # TS start before epoch, end during epoch not on TS point
    test_set(t, 16, 30, 5, 5)
    # TS start within epoch, end after epoch on TS point
    test_set(t, 0, 20, 0, 7)
    # TS start within epoch, end after epoch not on TS point
    test_set(t, 0, 18, 0, 5)
    # epoch within TS
    test_set(t, 15, 20, 4, 3)
    # epoch within TS with no overlap
    test_set(t, 16, 18, -1, 0)
    # epoch within TS with start between TS points and end on point
    test_set(t, 16, 21, 5, 3)
    # epoch within TS with start between TS points and end not on point
    test_set(t, 16, 23, 5, 4)
    # epoch within TS with start on TS point and end on point
    test_set(t, 15, 21, 4, 4)
    # epoch within TS with start on TS point and end not on point
    test_set(t, 15, 23, 4, 5)
    print "EpochTS self-test OK"

def test_set(t, e0, e1, i0, cnt):
    """Debug code -- tests interval overlap detection/calculation
    """
    print "** Testing epoch (%d, %d)" % (e0, e1)
    idx = find_ts_interval_start(t, e0, e1)
    assert idx == i0, "Wanted idx %d, got %d" % (i0, idx)
    if i0 >= 0:
        c = find_ts_interval_overlap(i0, t, e0, e1)
        assert c == cnt, "Wanted cnt %d, got %d" % (cnt, c)

def find_ts_interval(t, epoch_start, epoch_stop):
    """ Finds the overlapping section of array *t* with epoch start
        and stop times.

        Args:
            *t* (double array) Timestamp array
            *epoch_start* (double) Epoch start time
            *epoch_stop* (double) Epoch stop time

        Returns:
            *idx* (int) Index of first element in *t* at or after *epoch_start*
            *count* (int) Number of elements in *t* between *epoch_start* and *epoch_stop*
    """
    idx = find_ts_interval_start(t, epoch_start, epoch_stop)
    if idx < 0:
        return -1, -1
    cnt = find_ts_interval_overlap(idx, t, epoch_start, epoch_stop)
    return idx, cnt

def find_ts_interval_start(t, epoch_start, epoch_stop):
    """ Finds the first element in *t* that is >= *epoch_start*

        Args:
            *t* (double array) Timestamp array
            *epoch_start* (double) Epoch start time
            *epoch_stop* (double) Epoch stop time

        Returns:
            *idx* (int) Index of first element in *t* at or after *epoch_start*, or -1 if no overlap
    """
    i0 = 0
    t0 = t[i0]
    i1 = len(t) - 1
    t1 = t[i1]
    #print "Searching %d to %d" % (epoch_start, epoch_stop)
    # make sure there's overlap between epoch and time series
    # see if time series start is within range
    if t0 > epoch_stop or t1 < epoch_start:
        #print "No ts overlap"
        return -1
    # check edge case
    if epoch_start == t1:
        # epoch starts where time series ends
        # only overlap between timeseries and epoch is last sample
        #   of time series
        #print "epoch start == t1"
        return i1
    # else, start of epoch is somewhere in time series. look for it
    window = i1 - i0
    # search until we're within 4 samples, then search linearly
    while window > 4:
        mid = i0 + (i1 - i0) / 2
        midt = t[mid]
    #    print "%f\t%f\t%f\t(%d)" % (t0, midt, t1, window)
        if epoch_start == midt:
            #print "Found at mid=%d" % mid
            return mid
        if epoch_start > midt:
            # epoch start later than midpoint of window
            # move window start to midpoint and go again
            i0 = mid
            t0 = midt
        else:
            # start < midt: epoch start beforem midpoint of window
            #   so move window to end to midpoint and start again
            i1 = mid
            t1 = midt
        window = i1 - i0
    # sample is in narrow window. search linearly
    #print "Searching %d-%d (%f to %f)" % (i0, i1, t0, t1)
    for i in range(i0, i1+1):
        #print t[i]
        if t[i] >= epoch_start:
            # first sample greater than epoch start. make sure
            #   it's also before epoch end
            if t[i] < epoch_stop:
                #print "start<=%d<end, idx=%d" % (t[i], i)
                return i
            else:
                # epoch rests entirely between two timestamp 
                #    samples, with no overlap
                #print "No overlap"
                return -1
    #print "Epoch start: %f" % start_time
    #print "Epoch end: %f" % stop_time
    assert False, "Unable to find start of timeseries overlap with epoch"

def find_ts_interval_overlap(i0, t, epoch_start, epoch_stop):
    """ Finds the number of elements in *t* that overlap with epoch
        start/stop.

        Args:
            *i0* (int) Index of first element in *t* that is in [start,stop] interval
            *t* (double array) Timestamp array
            *epoch_start* (double) Epoch start time
            *epoch_stop* (double) Epoch stop time

        Returns:
            *cnt* (int) Number of elements in *t* that overlap with epoch
    """
    assert epoch_start <= t[i0]
    t0 = t[i0]
    i1 = len(t) - 1
    t1 = t[i1]
    # if we made it here, i0 is within epoch
    # see where timeseries ends relative to epoch end
    if t1 <= epoch_stop:
        # timeseries ends before or at end of epoch
        cnt = i1 - i0 + 1
        #print "\tA\t%d, %d -> %d" % (i0, i1, cnt)
        return cnt
    # time series extends beyond epoch end. find end of overlap
    # if we've made it here then there is at least one timeseries entry
    #   that extends beyond epoch
    for i in range(i0, (len(t)-1)):
        if t[i] <= epoch_stop and t[i+1] > epoch_stop:
            cnt = i - i0 + 1
            #print "\tB\t%d, %d -> %d" % (i0, i, cnt)
            return cnt
    assert False, "Failed to find overlap"


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
        if nwb is not None:
            if name in self.nwb.file_pointer["epochs"]:
                nwb.fatal_error("Duplicate epoch names ('%s')" % name)
        else:
            print "** Warning: creating epoch without specifying neurodata file"
            print "Hopefully this is because of an EpochTS self-test"
        # go ahead and create epoch folder now
        fp = nwb.file_pointer
        epoch = fp["epochs"].create_group(self.name)
        self.finalized = False

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
            print "** Error **"
            print "Time series '%s' not found" % timeseries_path
            assert False
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
        e0 = self.start_time
        e1 = self.stop_time
        start_idx = find_ts_interval_start(t, e0, e1)
        epoch_ts["start_idx"] = start_idx
        if start_idx < 0:
            #sys.stderr.write("\t%s has no data in %s\n" % (in_epoch_name, self.name))
            return
        cnt = find_ts_interval_overlap(start_idx, t, e0, e1)
        epoch_ts["count"] = cnt
        if cnt <= 0:
            #sys.stderr.write("\t%s has no overlap with %s\n" % (in_epoch_name, self.name))
            return
        self.timeseries_dict[in_epoch_name] = epoch_ts

    def set_description(self, desc):
        """ This sets the epoch's description field (a required field
            but that can be empty)

            Args:
                *desc* (text) Textual description 

            Returns:
                *nothing*
        """
        self.spec["description"]["_value"] = desc
        
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
        import nwb
        nwb.write_json("out.json", self.spec)
        self.nwb.write_datasets(grp, "", self.spec, False, False)
        self.nwb.write_attributes(grp, self.spec)
        # flag ourself as done
        self.finalized = True

