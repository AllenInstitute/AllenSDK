import pytz
from datetime import datetime
import pandas as pd
from typing import Optional

from allensdk.internal.api import (
    OneOrMoreResultExpectedError, db_connection_creator)
from allensdk.api.warehouse_cache.cache import memoize
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin
from allensdk.core.authentication import DbCredentials
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


class OphysLimsExtractor(CachedInstanceMethodMixin):
    """A data fetching class that serves as an API for fetching 'raw'
    data from LIMS for filling optical physiology data. This data is
    is necessary (but not sufficient) to fill the 'Ophys' portion of a
    BehaviorOphysExperiment.

    This class needs to be inherited by the BehaviorOphysLimsApi and also
    have methods from BehaviorOphysDataTransforms in order to be usable by a
    BehaviorOphysExperiment.
    """

    def __init__(self, ophys_experiment_id: int,
                 lims_credentials: Optional[DbCredentials] = None):
        self.ophys_experiment_id = ophys_experiment_id

        self.lims_db = db_connection_creator(
            credentials=lims_credentials,
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

    def get_ophys_experiment_id(self):
        return self.ophys_experiment_id

    @memoize
    def get_plane_group_count(self) -> int:
        """Gets the total number of plane groups in the session.
        This is required for resampling ophys timestamps for mesoscope
        data. Will be 0 if the scope did not capture multiple concurrent
        frames. See `get_imaging_plane_group` for more info.
        """
        query = f"""
            -- Get the session ID for an experiment
            WITH sess AS (
                SELECT os.id from ophys_experiments oe
                JOIN ophys_sessions os ON os.id = oe.ophys_session_id
                WHERE oe.id = {self.ophys_experiment_id}
            )
            SELECT  COUNT(DISTINCT(pg.group_order)) AS planes
            FROM  ophys_sessions os
            JOIN ophys_experiments oe ON os.id = oe.ophys_session_id
            JOIN  ophys_imaging_plane_groups pg
                ON pg.id = oe.ophys_imaging_plane_group_id
            WHERE
                -- only 1 session for an experiment
                os.id = (SELECT id from sess limit 1);
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_imaging_plane_group(self) -> Optional[int]:
        """Get the imaging plane group number. This is a numeric index
        that indicates the order that the frames were acquired when
        there is more than one frame acquired concurrently. Relevant for
        mesoscope data timestamps, as the laser jumps between plane
        groups during the scan. Will be None for non-mesoscope data.
        """
        query = f"""
            SELECT pg.group_order
            FROM ophys_experiments oe
            JOIN ophys_imaging_plane_groups pg
            ON pg.id = oe.ophys_imaging_plane_group_id
            WHERE oe.id = {self.get_ophys_experiment_id()};
        """
        # Non-mesoscope data will not have results
        group_order = self.lims_db.fetchall(query)
        if len(group_order):
            return group_order[0]
        else:
            return None

    @memoize
    def get_behavior_session_id(self) -> Optional[int]:
        """Returns the behavior_session_id associated with this experiment,
        if applicable.
        """
        query = f"""
            SELECT bs.id
            FROM ophys_experiments oe
            -- every ophys_experiment should have an ophys_session
            JOIN ophys_sessions os ON oe.ophys_session_id = os.id
            -- but not every ophys_session has a behavior_session
            LEFT JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            WHERE oe.id = {self.get_ophys_experiment_id()};
        """
        response = self.lims_db.fetchall(query)     # Can be null
        if not len(response):
            return None
        else:
            return response[0]

    @memoize
    def get_ophys_experiment_dir(self) -> str:
        """Get the storage directory associated with the ophys experiment"""
        query = """
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_nwb_filepath(self) -> str:
        """Get the filepath of the nwb file associated with the ophys
        experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkft.name = 'NWBOphys'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_sync_file(self, ophys_experiment_id=None) -> str:
        """Get the filepath of the sync timing file associated with the
        ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN well_known_files wkf ON wkf.attachable_id = os.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkf.attachable_type = 'OphysSession'
                AND wkft.name = 'OphysRigSync'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_max_projection_file(self) -> str:
        """Get the filepath of the max projection image associated with the
        ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS maxint_file
                FROM ophys_experiments oe
                JOIN ophys_cell_segmentation_runs ocsr
                ON ocsr.ophys_experiment_id = oe.id
                JOIN well_known_files wkf ON wkf.attachable_id = ocsr.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE ocsr.current = 't'
                AND wkf.attachable_type = 'OphysCellSegmentationRun'
                AND wkft.name = 'OphysMaxIntImage'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_targeted_structure(self) -> str:
        """Get the targeted structure (acronym) for an ophys experiment
        (ex: "Visp")"""
        query = """
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id = oe.targeted_structure_id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_imaging_depth(self) -> int:
        """Get the imaging depth for an ophys experiment
        (ex: 400, 500, etc.)"""
        query = """
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_stimulus_name(self) -> str:
        """Get the name of the stimulus presented for an ophys experiment"""
        query = """
                SELECT os.stimulus_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        stimulus_name = self.lims_db.fetchone(query, strict=False)
        stimulus_name = 'Unknown' if stimulus_name is None else stimulus_name
        return stimulus_name

    @memoize
    def get_date_of_acquisition(self) -> datetime:
        """Get the acquisition date of an ophys experiment"""
        query = """
                SELECT os.date_of_acquisition
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())

        experiment_date = self.lims_db.fetchone(query, strict=True)
        return pytz.utc.localize(experiment_date)

    @memoize
    def get_reporter_line(self) -> str:
        """Get the (gene) reporter line for the subject associated with an
        ophys experiment
        """
        query = """
                SELECT g.name as reporter_line
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id = os.specimen_id
                JOIN donors d ON d.id = sp.donor_id
                JOIN donors_genotypes dg ON dg.donor_id = d.id
                JOIN genotypes g ON g.id = dg.genotype_id
                JOIN genotype_types gt ON gt.id = g.genotype_type_id
                WHERE gt.name = 'reporter'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' from query")
        return result

    @memoize
    def get_driver_line(self) -> str:
        """Get the (gene) driver line for the subject associated with an ophys
        experiment"""
        query = """
                SELECT g.name as driver_line
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id = os.specimen_id
                JOIN donors d ON d.id = sp.donor_id
                JOIN donors_genotypes dg ON dg.donor_id = d.id
                JOIN genotypes g ON g.id = dg.genotype_id
                JOIN genotype_types gt ON gt.id = g.genotype_type_id
                WHERE gt.name = 'driver'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' from query")
        return result

    @memoize
    def get_external_specimen_name(self) -> int:
        """Get the external specimen id (LabTracks ID) for the subject
        associated with an ophys experiment"""
        query = """
                SELECT sp.external_specimen_name
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id = os.specimen_id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return int(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_full_genotype(self) -> str:
        """Get the full genotype of the subject associated with an ophys
        experiment"""
        query = """
                SELECT d.full_genotype
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id = os.specimen_id
                JOIN donors d ON d.id = sp.donor_id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_dff_file(self) -> str:
        """Get the filepath of the dff trace file associated with an ophys
        experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS dff_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkft.name = 'OphysDffTraceFile'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_objectlist_file(self) -> str:
        """Get the objectlist.txt filepath associated with an ophys experiment

        NOTE: Although this will be deprecated for visual behavior it will
        still be valid for visual coding.
        """
        query = """
                SELECT wkf.storage_directory || wkf.filename AS obj_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr
                ON ocsr.ophys_experiment_id = oe.id
                JOIN well_known_files wkf ON wkf.attachable_id = ocsr.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkft.name = 'OphysSegmentationObjects'
                AND wkf.attachable_type = 'OphysCellSegmentationRun'
                AND ocsr.current = 't'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_demix_file(self) -> str:
        """Get the filepath of the demixed traces file associated with an
        ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS demix_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkf.attachable_type = 'OphysExperiment'
                AND wkft.name = 'DemixedTracesFile'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_average_intensity_projection_image_file(self) -> str:
        """Get the avg intensity project image filepath associated with an
        ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS avgint_file
                FROM ophys_experiments oe
                JOIN ophys_cell_segmentation_runs ocsr
                ON ocsr.ophys_experiment_id = oe.id
                JOIN well_known_files wkf ON wkf.attachable_id=ocsr.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE ocsr.current = 't'
                AND wkf.attachable_type = 'OphysCellSegmentationRun'
                AND wkft.name = 'OphysAverageIntensityProjectionImage'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_rigid_motion_transform_file(self) -> str:
        """Get the filepath for the motion transform file (.csv) associated
        with an ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS transform_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkft.id = wkf.well_known_file_type_id
                WHERE wkf.attachable_type = 'OphysExperiment'
                AND wkft.name = 'OphysMotionXyOffsetData'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_motion_corrected_image_stack_file(self) -> str:
        """Get the filepath for the motion corrected image stack associated
        with a an ophys experiment"""
        query = """
            SELECT wkf.storage_directory || wkf.filename
            FROM well_known_files wkf
            JOIN well_known_file_types wkft
            ON wkft.id = wkf.well_known_file_type_id
            WHERE wkft.name = 'MotionCorrectedImageStack'
            AND wkf.attachable_id = {};
            """.format(self.get_ophys_experiment_id())

        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_foraging_id(self) -> str:
        """Get the foraging id associated with an ophys experiment. This
        id is obtained in str format but can be interpreted as a UUID.
        (ex: 6448125b-5d18-4bda-94b6-fb4eb6613979)"""
        query = """
                SELECT os.foraging_id
                FROM ophys_experiments oe
                LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                WHERE oe.id= {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_field_of_view_shape(self) -> dict:
        """Get a field of view dictionary for a given ophys experiment.
           ex: {"width": int, "height": int}
        """
        query = """
                SELECT {}
                FROM ophys_experiments oe
                WHERE oe.id = {};
                """

        fov_shape = dict()
        ophys_expt_id = self.get_ophys_experiment_id()
        for dim in ['width', 'height']:
            select_col = f'oe.movie_{dim}'
            formatted_query = query.format(select_col, ophys_expt_id)
            fov_shape[dim] = self.lims_db.fetchone(formatted_query,
                                                   strict=True)
        return fov_shape

    @memoize
    def get_ophys_cell_segmentation_run_id(self) -> int:
        """Get the ophys cell segmentation run id associated with an
        ophys experiment id"""
        query = """
                SELECT oseg.id
                FROM ophys_experiments oe
                JOIN ophys_cell_segmentation_runs oseg
                ON oe.id = oseg.ophys_experiment_id
                WHERE oseg.current = 't'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_raw_cell_specimen_table_dict(self) -> dict:
        """Get the cell_rois table from LIMS in dictionary form"""
        ophys_cell_seg_run_id = self.get_ophys_cell_segmentation_run_id()
        query = """
                SELECT *
                FROM cell_rois cr
                WHERE cr.ophys_cell_segmentation_run_id = {};
                """.format(ophys_cell_seg_run_id)
        initial_cs_table = pd.read_sql(query, self.lims_db.get_connection())
        cell_specimen_table = initial_cs_table.rename(
            columns={'id': 'cell_roi_id', 'mask_matrix': 'roi_mask'})
        cell_specimen_table.drop(['ophys_experiment_id',
                                  'ophys_cell_segmentation_run_id'],
                                 inplace=True, axis=1)
        return cell_specimen_table.to_dict()

    @memoize
    def get_surface_2p_pixel_size_um(self) -> float:
        """Get the pixel size for 2-photon movies in micrometers"""
        query = """
                SELECT sc.resolution
                FROM ophys_experiments oe
                JOIN scans sc ON sc.image_id=oe.ophys_primary_image_id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_workflow_state(self) -> str:
        """Get the workflow state of an ophys experiment (ex: 'failed')"""
        query = """
                SELECT oe.workflow_state
                FROM ophys_experiments oe
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        query = """
                SELECT g.name as sex
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN genders g ON g.id=d.gender_id
                WHERE oe.id= {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_age(self) -> str:
        """Get the age of the subject (ex: 'P15', 'Adult', etc...)"""
        query = """
                SELECT a.name as age
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id
                JOIN ages a ON a.id=d.age_id
                WHERE oe.id= {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)


if __name__ == "__main__":

    api = OphysLimsExtractor(789359614)
    print(api.get_age())
