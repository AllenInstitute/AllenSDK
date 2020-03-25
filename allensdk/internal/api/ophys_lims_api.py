import matplotlib.image as mpimg  # NOQA: D102
import json
import numpy as np
import os
import h5py
import pytz
import pandas as pd
from typing import Optional

from allensdk.internal.api import (
    PostgresQueryMixin, OneOrMoreResultExpectedError)
from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.image_api import ImageApi
import allensdk.brain_observatory.roi_masks as roi
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin
from allensdk.core.authentication import credential_injector, DbCredentials
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


class OphysLimsApi(CachedInstanceMethodMixin):

    def __init__(self, ophys_experiment_id: int,
                 lims_credentials: Optional[DbCredentials] = None):
        self.ophys_experiment_id = ophys_experiment_id
        if lims_credentials:
            self.lims_db = PostgresQueryMixin(
                dbname=lims_credentials.dbname, user=lims_credentials.user,
                host=lims_credentials.host, password=lims_credentials.password,
                port=lims_credentials.port)
        else:
            # Currying is equivalent to decorator syntactic sugar
            self.lims_db = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
                            (PostgresQueryMixin)())

    def get_ophys_experiment_id(self):
        return self.ophys_experiment_id

    @memoize
    def get_ophys_experiment_dir(self):
        query = '''
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_nwb_filepath(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'NWBOphys')
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_sync_file(self, ophys_experiment_id=None):
        query = '''
                SELECT sync.storage_directory || sync.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_max_projection_file(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS maxint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=ocsr.id AND wkf.attachable_type = 'OphysCellSegmentationRun' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMaxIntImage')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_max_projection(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        maxInt_a13_file = self.get_max_projection_file()
        pixel_size = self.get_surface_2p_pixel_size_um()
        max_projection = mpimg.imread(maxInt_a13_file)
        return ImageApi.serialize(max_projection, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    @memoize
    def get_targeted_structure(self):
        query = '''
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id=oe.targeted_structure_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_imaging_depth(self):
        query = '''
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN imaging_depths id ON id.id=oe.imaging_depth_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_stimulus_name(self):
        query = '''
                SELECT os.stimulus_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        stimulus_name = self.lims_db.fetchone(query, strict=False)
        stimulus_name = 'Unknown' if stimulus_name is None else stimulus_name
        return stimulus_name


    @memoize
    def get_experiment_date(self):
        query = '''
                SELECT os.date_of_acquisition
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())

        experiment_date = self.lims_db.fetchone(query, strict=True)
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
                '''.format(self.get_ophys_experiment_id())
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError('Expected one or more, but received: {} from query'.format(result))
        return result

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
                '''.format(self.get_ophys_experiment_id())
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError('Expected one or more, but received: {} from query'.format(result))
        return result

    @memoize
    def get_external_specimen_name(self, ophys_experiment_id=None):
        query = '''
                SELECT sp.external_specimen_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return int(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_full_genotype(self):
        query = '''
                SELECT d.full_genotype
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_equipment_id(self):
        query = '''
                SELECT e.name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN equipment e ON e.id=os.equipment_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_dff_file(self):
        query = '''
                SELECT dff.storage_directory || dff.filename AS dff_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files dff ON dff.attachable_id=oe.id AND dff.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysDffTraceFile')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_cell_roi_ids(self):
        cell_specimen_table = self.get_cell_specimen_table()
        assert cell_specimen_table.index.name == 'cell_specimen_id'
        return cell_specimen_table['cell_roi_id'].values

    @memoize
    def get_objectlist_file(self):
        query = '''
                SELECT obj.storage_directory || obj.filename AS obj_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_demix_file(self):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS demix_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.attachable_type = 'OphysExperiment' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'DemixedTracesFile')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_average_intensity_projection_image_file(self):
        query = '''
                SELECT avg.storage_directory || avg.filename AS avgint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files avg ON avg.attachable_id=ocsr.id AND avg.attachable_type = 'OphysCellSegmentationRun' AND avg.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysAverageIntensityProjectionImage')
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_rigid_motion_transform_file(self):
        query = '''
                SELECT tra.storage_directory || tra.filename AS transform_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files tra ON tra.attachable_id=oe.id AND tra.attachable_type = 'OphysExperiment' AND tra.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMotionXyOffsetData')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_motion_corrected_image_stack_file(self):
        query = f"""
            select wkf.storage_directory || wkf.filename
            from well_known_files wkf
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkft.name = 'MotionCorrectedImageStack'
            and wkf.attachable_id = {self.get_ophys_experiment_id()}
        """
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_foraging_id(self):
        query = '''
                SELECT os.foraging_id
                FROM ophys_experiments oe
                LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())        
        return self.lims_db.fetchone(query, strict=True)

    def get_raw_dff_data(self):
        dff_path = self.get_dff_file()
        with h5py.File(dff_path, 'r') as raw_file:
            dff_traces = np.asarray(raw_file['data'])
        return dff_traces

    @memoize
    def get_rig_name(self):
        query = '''
                select e.name as device_name
                from ophys_experiments oe
                join ophys_sessions os on os.id = oe.ophys_session_id
                join equipment e on e.id = os.equipment_id
                where oe.id = {}
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_field_of_view_shape(self):
        query = '''
                select {}
                from ophys_experiments oe
                where oe.id = {}
                '''
        X = {c: self.lims_db.fetchone(query.format('oe.movie_{}'.format(c), self.get_ophys_experiment_id()), strict=True) for c in ['width', 'height']}
        return X

    @memoize
    def get_metadata(self):

        metadata = {}
        metadata['rig_name'] = self.get_rig_name()
        metadata['sex'] = self.get_sex()
        metadata['age'] = self.get_age()
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
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_raw_cell_specimen_table_dict(self):
        ophys_cell_segmentation_run_id = self.get_ophys_cell_segmentation_run_id()
        query = '''
                select *
                from cell_rois cr
                where cr.ophys_cell_segmentation_run_id = {}
                '''.format(ophys_cell_segmentation_run_id)
        cell_specimen_table = pd.read_sql(query, self.lims_db.get_connection()).rename(columns={'id': 'cell_roi_id', 'mask_matrix': 'image_mask'})
        cell_specimen_table.drop(['ophys_experiment_id', 'ophys_cell_segmentation_run_id'], inplace=True, axis=1)
        return cell_specimen_table.to_dict()

    @memoize
    def get_cell_specimen_table(self):
        cell_specimen_table = pd.DataFrame.from_dict(self.get_raw_cell_specimen_table_dict()).set_index('cell_roi_id').sort_index()
        fov_width, fov_height = self.get_field_of_view_shape()['width'], self.get_field_of_view_shape()['height']
        image_mask_list = []
        for sub_mask in cell_specimen_table['image_mask'].values:
            curr_roi = roi.create_roi_mask(fov_width, fov_height, [(fov_width - 1), 0, (fov_height - 1), 0], roi_mask=np.array(sub_mask, dtype=np.bool))
            image_mask_list.append(curr_roi.get_mask_plane().astype(np.bool))
        cell_specimen_table['image_mask'] = image_mask_list
        cell_specimen_table = cell_specimen_table[sorted(cell_specimen_table.columns)]

        cell_specimen_table.index.rename('cell_roi_id', inplace=True)
        cell_specimen_table.reset_index(inplace=True)
        cell_specimen_table.set_index('cell_specimen_id', inplace=True)
        return cell_specimen_table

    @memoize
    def get_surface_2p_pixel_size_um(self):
        query = '''
                SELECT sc.resolution
                FROM ophys_experiments oe
                JOIN scans sc ON sc.image_id=oe.ophys_primary_image_id
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)


    @memoize
    def get_workflow_state(self):
        query = '''
                SELECT oe.workflow_state
                FROM ophys_experiments oe
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_segmentation_mask_image_file(self):
        query = '''
                SELECT obj.storage_directory || obj.filename AS OphysSegmentationMaskImage_filename
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationMaskImage')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))


    @memoize
    def get_segmentation_mask_image(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        segmentation_mask_image_file = self.get_segmentation_mask_image_file()
        pixel_size = self.get_surface_2p_pixel_size_um()
        segmentation_mask_image = mpimg.imread(segmentation_mask_image_file)
        return ImageApi.serialize(segmentation_mask_image, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    @memoize
    def get_sex(self):
        query = '''
                SELECT g.name as sex
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN genders g ON g.id=d.gender_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_age(self):
        query = '''
                SELECT a.name as age
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN ages a ON a.id=d.age_id
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)


if __name__ == "__main__":

    api = OphysLimsApi(789359614)
    print(api.get_age())
