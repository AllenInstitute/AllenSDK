import numpy as np
from .lims_api import LimsApi, clean_multiline_query


class OphysLimsApi(LimsApi):
    def __init__(self):
        super(OphysLimsApi, self).__init__()
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

    def session_id(self, experiment_id):
        session_query = clean_multiline_query('''
            select ophys_session_id
            from ophys_experiments
            where id = {}
        '''.format(experiment_id))
        df = self.query_fn(session_query)
        return df.loc[0, 'ophys_session_id']

    def get_sync_file(self, experiment_id):
        session_id = self.session_id(experiment_id)
        return self._get_well_known_file_path(session_id, 'OphysRigSync',
                                              'OphysSession')

    def get_pickle_file(self, experiment_id):
        session_id = self.session_id(experiment_id)
        return self._get_well_known_file_path(session_id, 'StimulusPickle',
                                              'OphysSession')

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

    def get_session_metadata(self, experiment_id):
        query = clean_multiline_query('''
            select oe.id as experiment_id, os.id as session_id, e.name as device_name,
            idepth.depth as imaging_depth_um, s.acronym as targeted_structure,
            os.date_of_acquisition as session_start_time, os.stimulus_name as session_description
            from ophys_experiments oe
            join ophys_sessions os on os.id = oe.ophys_session_id
            join equipment e on e.id = os.equipment_id
            join imaging_depths idepth on idepth.id = oe.imaging_depth_id
            join structures s on s.id = oe.targeted_structure_id
            where oe.id = {}
        '''.format(experiment_id))

        df = self.query_fn(query)
        metadata = df.loc[0].to_dict()
        # hardcode some stuff until I can figure out where to get it
        metadata['excitation_lambda'] = 910.
        metadata['emission_lambda'] = 520.
        metadata['indicator'] = 'GCAMP6f'
        metadata['imaging_rate'] = '31.'
        metadata['fov'] = '400x400 microns (512 x 512 pixels)'

        return metadata