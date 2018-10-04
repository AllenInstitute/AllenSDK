import numpy as np
from .lims_api import LimsApi, clean_multiline_query


class OphysApi(LimsApi):
    def __init__(self):
        super(OphysApi, self).__init__()
        self._dff_table = None

    def get_dff_file_table(self):
        # Dff files are apparently missing attachable type
        if self._dff_table is None:
            self._dff_table = self.get_well_known_file_table(
                well_known_file_type_names=['OphysDffTraceFile'])

        return self._dff_table

    def get_dff_file(self, experiment_id):
        table = self.get_dff_file_table()
        return table[table.attachable_id == experiment_id].iloc[0].path

    def get_roi_table(self, experiment_id):
        query = clean_multiline_query('''
            select cr.id, cr.cell_specimen_id, oe.movie_width, oe.movie_height, cr.x, cr.y, cr.mask_matrix
            from cell_rois cr
            join ophys_cell_segmentation_runs ocsr on ocsr.id = cr.ophys_cell_segmentation_run_id
            join ophys_experiments oe on oe.id = cr.ophys_experiment_id
            where cr.valid_roi = true and ocsr.current = true
            and oe.id = {}
        '''.format(experiment_id))
        
        df = self.query_fn(query)
        df.set_index('id')

        return df

    def get_roi_dict(self, experiment_id, key_by_cell_specimen=True):
        key = "cell_specimen_id" if key_by_cell_specimen else "id"

        table = self.get_roi_table(experiment_id)
        roi_dict = {}

        for _, row in table.iterrows():
            mask = np.zeros((row.movie_height, row.movie_width), dtype=bool)
            roi_mask = np.array(row.mask_matrix)
            h, w = roi_mask.shape
            mask[row.y:row.y + h, row.x:row.x + w] = roi_mask
            roi_dict[str(row[key])] = mask

        return roi_dict