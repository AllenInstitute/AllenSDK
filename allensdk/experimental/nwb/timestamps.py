import logging
from allensdk.experimental.aibs_sync.dataset import Dataset

logger = logging.getLogger(__name__)


def get_timestamps_from_sync(sync_file, key, use_falling_edges=True):
    dset = Dataset(sync_file)
    if use_falling_edges:
        result = dset.get_falling_edges(key, units="seconds")
    else:
        result = dset.get_rising_edges(key, units="seconds")

    return result


def correct_timestamps_length(times, data_length):
    if len(times) < data_length:
        raise ValueError(
            "Invalid timestamps length {} for data length {}".format(
                len(times), data_length
            )
        )
    elif len(times) > data_length:
        logger.warning(
            "Got timestamps length %s, data length %s, truncating timestamps",
            len(times),
            data_length
        )
        times = times[:data_length]

    return times
