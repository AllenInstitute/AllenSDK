from .core import get_receptive_field_data_dict_with_postprocessing
import numpy as np

def get_receptive_fields(lsn):

    metadata = lsn.data_set.get_metadata()

    oeid = metadata['ophys_experiment_id']
    csid_list = lsn.data_set.get_cell_specimen_ids()

    receptive_fields = np.empty((lsn.nrows, lsn.ncols, lsn.numbercells + 1, 2))

    for csid in csid_list:
        receptive_field_data_dict = get_receptive_field_data_dict_with_postprocessing(lsn.data_set, csid, alpha=.05)

    return receptive_fields