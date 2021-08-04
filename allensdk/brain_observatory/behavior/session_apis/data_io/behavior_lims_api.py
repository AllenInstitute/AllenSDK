import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
import pytz

from allensdk.api.warehouse_cache.cache import memoize
from allensdk.brain_observatory.behavior.session_apis.abcs.\
    data_extractor_base.behavior_data_extractor_base import \
    BehaviorDataExtractorBase
from allensdk.brain_observatory.behavior.session_apis.data_transforms import \
    BehaviorDataTransforms
from allensdk.core.auth_config import (LIMS_DB_CREDENTIAL_MAP,
                                       MTRAIN_DB_CREDENTIAL_MAP)
from allensdk.core.authentication import DbCredentials
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin
from allensdk.internal.api import (OneOrMoreResultExpectedError,
                                   OneResultExpectedError,
                                   db_connection_creator)
from allensdk.internal.core.lims_utilities import safe_system_path


class BehaviorLimsApi(BehaviorDataTransforms, CachedInstanceMethodMixin):
    """A data fetching and processing class that serves processed data from
    a specified raw data source (extractor). Contains all methods
    needed to fill a BehaviorSession."""

    def __init__(self,
                 behavior_session_id: Optional[int] = None,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None,
                 extractor: Optional[BehaviorDataExtractorBase] = None):

        if extractor is None:
            if behavior_session_id is not None:
                extractor = BehaviorLimsExtractor(
                    behavior_session_id,
                    lims_credentials,
                    mtrain_credentials)
            else:
                raise RuntimeError(
                    "BehaviorLimsApi must be provided either an instantiated "
                    "'extractor' or a 'behavior_session_id'!")

        super().__init__(extractor=extractor)


class BehaviorLimsExtractor(BehaviorDataExtractorBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from LIMS necessary (but not sufficient) for filling a
    'BehaviorSession'.

    Most 'raw' data provided by this API needs to be processed by
    BehaviorDataTransforms methods in order to usable by 'BehaviorSession's
    """
    def __init__(self, behavior_session_id: int,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.mtrain_db = db_connection_creator(
            credentials=mtrain_credentials,
            fallback_credentials=MTRAIN_DB_CREDENTIAL_MAP)

        self.lims_db = db_connection_creator(
            credentials=lims_credentials,
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

        self.behavior_session_id = behavior_session_id
        ids = self._get_ids()
        self.ophys_experiment_ids = ids.get("ophys_experiment_ids")
        self.ophys_session_id = ids.get("ophys_session_id")
        self.foraging_id = ids.get("foraging_id")
        self.ophys_container_id = ids.get("ophys_container_id")

    @classmethod
    def from_foraging_id(cls,
                         foraging_id: Union[str, uuid.UUID, int],
                         lims_credentials: Optional[DbCredentials] = None
                         ) -> "BehaviorLimsApi":
        """Create a BehaviorLimsAPI instance from a foraging_id instead of
        a behavior_session_id.

        NOTE: 'foraging_id' in the LIMS behavior_session table should be
              the same as the 'behavior_session_uuid' in mtrain which should
              also be the same as the 'session_uuid' field in the .pkl
              returned by 'get_behavior_stimulus_file()'.
        """

        lims_db = db_connection_creator(
            credentials=lims_credentials,
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

        if isinstance(foraging_id, uuid.UUID):
            foraging_id = str(foraging_id)
        elif isinstance(foraging_id, int):
            foraging_id = str(uuid.UUID(int=foraging_id))

        query = f"""
            SELECT id
            FROM behavior_sessions
            WHERE foraging_id = '{foraging_id}';
        """
        session_id = lims_db.fetchone(query, strict=True)
        return cls(session_id, lims_credentials=lims_credentials)

    def _get_ids(self) -> Dict[str, Optional[Union[int, List[int]]]]:
        """Fetch ids associated with this behavior_session_id. If there is no
        id, return None.
        :returns: Dictionary of ids with the following keys:
            ophys_session_id: int
            ophys_experiment_ids: List[int] -- only if have ophys_session_id
            foraging_id: int
        :rtype: dict
        """
        # Get all ids from the behavior_sessions table
        query = f"""
            SELECT
                ophys_session_id, foraging_id
            FROM
                behavior_sessions
            WHERE
                behavior_sessions.id = {self.behavior_session_id};
        """
        ids_response = self.lims_db.select(query)
        if len(ids_response) > 1 or len(ids_response) < 1:
            raise OneResultExpectedError(
                f"Expected length one result, received: "
                f"{ids_response} results from query")
        ids_dict = ids_response.iloc[0].to_dict()

        #  Get additional ids if also an ophys session
        #     (experiment_id, container_id)
        if ids_dict.get("ophys_session_id"):
            oed_query = f"""
                SELECT id
                FROM ophys_experiments
                WHERE ophys_session_id = {ids_dict["ophys_session_id"]};
                """
            oed = self.lims_db.fetchall(oed_query)
            if len(oed) == 0:
                oed = None

            container_query = f"""
            SELECT DISTINCT
                visual_behavior_experiment_container_id id
            FROM
                ophys_experiments_visual_behavior_experiment_containers
            WHERE
                ophys_experiment_id IN ({",".join(set(map(str, oed)))});
            """
            try:
                container_id = self.lims_db.fetchone(container_query,
                                                     strict=True)
            except OneResultExpectedError:
                container_id = None

            ids_dict.update({"ophys_experiment_ids": oed,
                             "ophys_container_id": container_id})
        else:
            ids_dict.update({"ophys_experiment_ids": None,
                             "ophys_container_id": None})
        return ids_dict

    def get_behavior_session_id(self) -> int:
        """Getter to be consistent with BehaviorOphysLimsApi."""
        return self.behavior_session_id

    def get_ophys_experiment_ids(self) -> Optional[List[int]]:
        return self.ophys_experiment_ids

    def get_ophys_session_id(self) -> Optional[int]:
        return self.ophys_session_id

    def get_foraging_id(self) -> int:
        return self.foraging_id

    def get_ophys_container_id(self) -> Optional[int]:
        return self.ophys_container_id

    def get_behavior_stimulus_file(self) -> str:
        """Return the path to the StimulusPickle file for a session.
        :rtype: str
        """
        query = f"""
            SELECT
                stim.storage_directory || stim.filename AS stim_file
            FROM
                well_known_files stim
            WHERE
                stim.attachable_id = {self.behavior_session_id}
                AND stim.attachable_type = 'BehaviorSession'
                AND stim.well_known_file_type_id IN (
                    SELECT id
                    FROM well_known_file_types
                    WHERE name = 'StimulusPickle');
        """
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_birth_date(self) -> datetime:
        """Returns the birth date of the animal.
        :rtype: datetime.date
        """
        query = f"""
        SELECT d.date_of_birth
        FROM behavior_sessions bs
        JOIN donors d on d.id = bs.donor_id
        WHERE bs.id = {self.behavior_session_id};
        """
        return self.lims_db.fetchone(query, strict=True).date()

    @memoize
    def get_sex(self) -> str:
        """Returns sex of the animal (M/F)
        :rtype: str
        """
        query = f"""
            SELECT g.name AS sex
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id = d.id
            JOIN genders g ON g.id = d.gender_id
            WHERE bs.id = {self.behavior_session_id};
            """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_age(self) -> str:
        """Return the age code of the subject (ie P123)
        :rtype: str
        """
        query = f"""
            SELECT a.name AS age
            FROM behavior_sessions bs
            JOIN donors d ON d.id = bs.donor_id
            JOIN ages a ON a.id = d.age_id
            WHERE bs.id = {self.behavior_session_id};
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_equipment_name(self) -> str:
        """Returns the name of the experimental rig.
        :rtype: str
        """
        query = f"""
            SELECT e.name AS device_name
            FROM behavior_sessions bs
            JOIN equipment e ON e.id = bs.equipment_id
            WHERE bs.id = {self.behavior_session_id};
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_reporter_line(self) -> List[str]:
        """Returns the genotype name(s) of the reporter line(s).
        :rtype: list
        """
        query = f"""
            SELECT g.name AS reporter_line
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN donors_genotypes dg ON dg.donor_id=d.id
            JOIN genotypes g ON g.id=dg.genotype_id
            JOIN genotype_types gt
                ON gt.id=g.genotype_type_id AND gt.name = 'reporter'
            WHERE bs.id={self.behavior_session_id};
        """
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' "
                f"from query:\n'{query}'")
        return result

    @memoize
    def get_driver_line(self) -> List[str]:
        """Returns the genotype name(s) of the driver line(s).
        :rtype: list
        """
        query = f"""
            SELECT g.name AS driver_line
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN donors_genotypes dg ON dg.donor_id=d.id
            JOIN genotypes g ON g.id=dg.genotype_id
            JOIN genotype_types gt
                ON gt.id=g.genotype_type_id AND gt.name = 'driver'
            WHERE bs.id={self.behavior_session_id};
        """
        result = self.lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' "
                f"from query:\n'{query}'")
        return result

    @memoize
    def get_mouse_id(self) -> int:
        """Returns the LabTracks ID
        :rtype: int
        """
        # TODO: Should this even be included?
        # Found sometimes there were entries with NONE which is
        # why they are filtered out; also many entries in the table
        # match the donor_id, which is why used DISTINCT
        query = f"""
            SELECT DISTINCT(sp.external_specimen_name)
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN specimens sp ON sp.donor_id=d.id
            WHERE bs.id={self.behavior_session_id}
            AND sp.external_specimen_name IS NOT NULL;
            """
        return int(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_full_genotype(self) -> str:
        """Return the name of the subject's genotype
        :rtype: str
        """
        query = f"""
                SELECT d.full_genotype
                FROM behavior_sessions bs
                JOIN donors d ON d.id=bs.donor_id
                WHERE bs.id= {self.behavior_session_id};
                """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_date_of_acquisition(self) -> datetime:
        """Get the acquisition date of a behavior_session in UTC
        :rtype: datetime"""
        query = """
                SELECT bs.date_of_acquisition
                FROM behavior_sessions bs
                WHERE bs.id = {};
                """.format(self.behavior_session_id)

        experiment_date = self.lims_db.fetchone(query, strict=True)
        return pytz.utc.localize(experiment_date)
