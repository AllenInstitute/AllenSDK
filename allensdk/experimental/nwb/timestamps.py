# until we get a public reader...
# pip install --trusted-host aibs-artifactory --index-url http://aibs-artifactory/artifactory/api/pypi/pypi-local/simple aibs-sync-extractor
from aibs_sync_extractor.dataset import Dataset


def get_timestamps_from_sync(sync_file, key, use_falling_edges=True):
    dset = Dataset(sync_file)
    if use_falling_edges:
        result = dset.get_falling_edges(key, units="seconds")
    else:
        result = dset.get_rising_edges(key, units="seconds")

    return result
