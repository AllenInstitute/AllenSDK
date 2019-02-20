import matplotlib.image as mpimg  # NOQA: D102

from . import PostgresQueryMixin
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
                SELECT obj.storage_directory || 'maxInt_a13a.png' AS maxint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
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
            raise OneOrMoreResultExpectedError('Expected one or more, but received: {} results form query'.format(x))
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