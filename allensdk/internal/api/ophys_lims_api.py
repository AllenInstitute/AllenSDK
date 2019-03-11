import matplotlib.image as mpimg  # NOQA: D102
import json
import numpy as np
import os
import h5py

from . import PostgresQueryMixin, OneOrMoreResultExpectedError
from allensdk.api.cache import memoize

class OphysLimsApi(PostgresQueryMixin):
    ''' hello
    '''

    @memoize
    def get_ophys_experiment_dir(self, ophys_experiment_id=None):
        '''
        '''

        query = '''
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id = {ophys_experiment_id};
                '''.format(ophys_experiment_id=ophys_experiment_id)
        return self.fetchone(query, strict=True)


    @memoize
    def get_nwb_filepath(self, ophys_experiment_id=None):

        query = '''
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'NWBOphys')
                WHERE oe.id = {ophys_experiment_id};
                '''.format(ophys_experiment_id=ophys_experiment_id)
        return self.fetchone(query, strict=True)


    @memoize
    def get_sync_file(self, ophys_experiment_id=None):
        query = '''
                SELECT sync.storage_directory || sync.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_maxint_file(self, ophys_experiment_id=None):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS maxint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=ocsr.id AND wkf.attachable_type = 'OphysCellSegmentationRun' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMaxIntImage')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)
        return self.fetchone(query, strict=True)


    @memoize
    def get_max_projection(self, ophys_experiment_id=None):
        maxInt_a13_file = self.get_maxint_file(ophys_experiment_id=ophys_experiment_id)
        max_projection = mpimg.imread(maxInt_a13_file)
        return max_projection


    @memoize
    def get_sync_file(self, ophys_experiment_id=None):
        query = '''
                SELECT sync.storage_directory || sync.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_targeted_structure(self, ophys_experiment_id=None):
        query = '''
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id=oe.targeted_structure_id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_imaging_depth(self, ophys_experiment_id=None):
        query = '''
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_stimulus_name(self, ophys_experiment_id=None):
        query = '''
                SELECT os.stimulus_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_experiment_date(self, ophys_experiment_id=None):
        query = '''
                SELECT os.date_of_acquisition
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_reporter_line(self, ophys_experiment_id=None):
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
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_driver_line(self, ophys_experiment_id=None):
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
                '''.format(ophys_experiment_id)
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
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_full_genotype(self, ophys_experiment_id=None):
        query = '''
                SELECT d.full_genotype
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)

    @memoize
    def get_equipment_id(self, ophys_experiment_id=None):
        query = '''
                SELECT e.name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN equipment e ON e.id=os.equipment_id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_dff_file(self, ophys_experiment_id=None):
        query = '''
                SELECT dff.storage_directory || dff.filename AS dff_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files dff ON dff.attachable_id=oe.id AND dff.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysDffTraceFile')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_input_extract_traces_file(self, ophys_experiment_id=None):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS input_extract_traces_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.attachable_type = 'OphysExperiment' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysExtractedTracesInputJson')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_cell_roi_ids(self, ophys_experiment_id=None):
        input_extract_traces_file = self.get_input_extract_traces_file(ophys_experiment_id=ophys_experiment_id)
        with open(input_extract_traces_file, 'r') as w:
            jin = json.load(w)
        return np.array([roi['id'] for roi in jin['rois']])


    @memoize
    def get_objectlist_file(self, ophys_experiment_id=None):
        query = '''
                SELECT obj.storage_directory || obj.filename AS obj_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_demix_file(self, ophys_experiment_id=None):
        query = '''
                SELECT wkf.storage_directory || wkf.filename AS demix_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.attachable_type = 'OphysExperiment' AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'DemixedTracesFile')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_avgint_a1X_file(self, ophys_experiment_id=None):
        query = '''
                SELECT avg.storage_directory || avg.filename AS avgint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files avg ON avg.attachable_id=ocsr.id AND avg.attachable_type = 'OphysCellSegmentationRun' AND avg.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysAverageIntensityProjectionImage')
                WHERE oe.id = {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_rigid_motion_transform_file(self, ophys_experiment_id=None):
        query = '''
                SELECT tra.storage_directory || tra.filename AS transform_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files tra ON tra.attachable_id=oe.id AND tra.attachable_type = 'OphysExperiment' AND tra.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMotionXyOffsetData')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)


    @memoize
    def get_foraging_id(self, ophys_experiment_id=None):
        query = '''
                SELECT os.foraging_id
                FROM ophys_experiments oe
                LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=True)

    def get_raw_dff_data(self, ophys_experiment_id=None, use_acq_trigger=False):
        dff_path = self.get_dff_file(ophys_experiment_id=ophys_experiment_id)
        with h5py.File(dff_path, 'r') as raw_file:
            dff_traces = np.asarray(raw_file['data'])
        return dff_traces