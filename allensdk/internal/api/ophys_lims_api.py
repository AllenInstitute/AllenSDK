import matplotlib.image as mpimg  # NOQA: D102
import json
import numpy as np
import os
import h5py
import pytz
import pandas as pd

from . import PostgresQueryMixin, OneOrMoreResultExpectedError
from allensdk.api.cache import memoize
from allensdk.brain_observatory.image_api import ImageApi


class OphysLimsApi(PostgresQueryMixin):

    def __init__(self, ophys_experiment_id):
        self.ophys_experiment_id = ophys_experiment_id
        super().__init__()

    @memoize
    def get_ophys_experiment_dir(self):
        query = '''
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id = {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_nwb_filepath(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'NWBOphys')
                WHERE oe.id = {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_sync_file(self, ophys_experiment_id=None):
        query = '''
                SELECT sync.storage_directory || sync.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_maxint_file(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS maxint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=ocsr.id AND wkf.attachable_type = 'OphysCellSegmentationRun' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMaxIntImage')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_max_projection(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        maxInt_a13_file = self.get_maxint_file()
        platform_json_file = self.get_ophys_platform_json()
        platform_data = json.load(open(platform_json_file, 'r'))
        pixel_size = float(platform_data['registration']['surface_2p']['pixel_size_um'])
        max_projection = mpimg.imread(maxInt_a13_file)
        return ImageApi.serialize(max_projection, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    @memoize
    def get_targeted_structure(self):
        query = '''
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id=oe.targeted_structure_id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_imaging_depth(self):
        query = '''
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_stimulus_name(self):
        query = '''
                SELECT os.stimulus_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        stimulus_name = self.fetchone(query, strict=True)
        stimulus_name = 'Unknown' if stimulus_name is None else stimulus_name
        return stimulus_name


    @memoize
    def get_experiment_date(self):
        query = '''
                SELECT os.date_of_acquisition
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)

        experiment_date = self.fetchone(query, strict=True)
        return pytz.utc.localize(experiment_date)

    @memoize
    def get_reporter_line(self):
        query = '''
                SELECT g.name as reporter_line
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN donors_genotypes dg ON dg.donor_id=d.id
                JOIN genotypes g ON g.id=dg.genotype_id
                JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'reporter'
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_driver_line(self):
        query = '''
                SELECT g.name as driver_line
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN donors_genotypes dg ON dg.donor_id=d.id
                JOIN genotypes g ON g.id=dg.genotype_id
                JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'driver'
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        result = self.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError('Expected one or more, but received: {} from query'.format(result))
        return result

    @memoize
    def get_LabTracks_ID(self, ophys_experiment_id=None):
        query = '''
                SELECT sp.external_specimen_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return int(self.fetchone(query, strict=True))

    @memoize
    def get_full_genotype(self):
        query = '''
                SELECT d.full_genotype
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_equipment_id(self):
        query = '''
                SELECT e.name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN equipment e ON e.id=os.equipment_id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_dff_file(self):
        query = '''
                SELECT dff.storage_directory || dff.filename AS dff_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files dff ON dff.attachable_id=oe.id AND dff.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysDffTraceFile')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_input_extract_traces_file(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS input_extract_traces_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.attachable_type = 'OphysExperiment' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysExtractedTracesInputJson')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_cell_roi_ids(self):
        input_extract_traces_file = self.get_input_extract_traces_file()
        with open(input_extract_traces_file, 'r') as w:
            jin = json.load(w)
        return np.array([roi['id'] for roi in jin['rois']])

    @memoize
    def get_objectlist_file(self):
        query = '''
                SELECT obj.storage_directory || obj.filename AS obj_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_demix_file(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS demix_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.attachable_type = 'OphysExperiment' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'DemixedTracesFile')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_avgint_a1X_file(self):
        query = '''
                SELECT avg.storage_directory || avg.filename AS avgint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files avg ON avg.attachable_id=ocsr.id AND avg.attachable_type = 'OphysCellSegmentationRun' AND avg.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysAverageIntensityProjectionImage')
                WHERE oe.id = {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_rigid_motion_transform_file(self):
        query = '''
                SELECT tra.storage_directory || tra.filename AS transform_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files tra ON tra.attachable_id=oe.id AND tra.attachable_type = 'OphysExperiment' AND tra.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMotionXyOffsetData')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_foraging_id(self):
        query = '''
                SELECT os.foraging_id
                FROM ophys_experiments oe
                LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)        
        return self.fetchone(query, strict=True)

    def get_raw_dff_data(self):
        dff_path = self.get_dff_file()
        with h5py.File(dff_path, 'r') as raw_file:
            dff_traces = np.asarray(raw_file['data'])
        return dff_traces

    @memoize
    def get_ophys_platform_json(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS transform_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=os.id AND wkf.attachable_type = 'OphysSession' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysPlatformJson')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_device_name(self):
        query = '''
                select e.name as device_name
                from ophys_experiments oe
                join ophys_sessions os on os.id = oe.ophys_session_id
                join equipment e on e.id = os.equipment_id
                where oe.id = {}
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_field_of_view_shape(self):
        query = '''
                select {}
                from ophys_experiments oe
                where oe.id = {}
                '''
        X = {c: self.fetchone(query.format('oe.movie_{}'.format(c), self.ophys_experiment_id), strict=True) for c in ['width', 'height']}
        return X

    @memoize
    def get_metadata(self):

        metadata = {}
        metadata['device_name'] = self.get_device_name()
        metadata['excitation_lambda'] = 910.
        metadata['emission_lambda'] = 520.
        metadata['indicator'] = 'GCAMP6f'
        metadata['field_of_view_width'] = self.get_field_of_view_shape()['width']
        metadata['field_of_view_height'] = self.get_field_of_view_shape()['height']

        return metadata

    @memoize
    def get_ophys_cell_segmentation_run_id(self):
        query = '''
                select oseg.id
                from ophys_experiments oe
                join ophys_cell_segmentation_runs oseg on oe.id = oseg.ophys_experiment_id
                where oe.id = {} and oseg.current = 't'
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_cell_specimen_table(self):
        ophys_cell_segmentation_run_id = self.get_ophys_cell_segmentation_run_id()
        query = '''
                select *
                from cell_rois cr
                where cr.ophys_cell_segmentation_run_id = {}
                '''.format(ophys_cell_segmentation_run_id)

        cell_specimen_table = pd.read_sql(query, self.get_connection()).rename(columns={'id': 'cell_roi_id', 'mask_matrix': 'image_mask'}).set_index('cell_roi_id')

        return cell_specimen_table
