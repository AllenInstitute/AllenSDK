import numpy as np


def trim_discontiguous_times(times, threshold=100):
    times = np.array(times)
    intervals = np.diff(times)

    med_interval = np.median(intervals)
    interval_threshold = med_interval * threshold

    gap_indices = np.where(intervals > interval_threshold)[0]

    if len(gap_indices) == 0:
        return times

    return times[:gap_indices[0]+1]