import pandas as pd
from typing import Optional, List, Dict, Any, Iterable
import logging

from allensdk.brain_observatory.behavior.internal.behavior_project_base\
    import BehaviorProjectBase
from allensdk.brain_observatory.behavior.behavior_data_session import (
    BehaviorDataSession)
from allensdk.brain_observatory.behavior.behavior_ophys_session import (
    BehaviorOphysSession)
from allensdk.internal.api.behavior_data_lims_api import BehaviorDataLimsApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.ecephys.ecephys_project_api.http_engine import (
    HttpEngine)
from allensdk.core.typing import SupportsStr
from allensdk.core.authentication import DbCredentials, credential_injector
from allensdk.core.auth_config import (
    MTRAIN_DB_CREDENTIAL_MAP, LIMS_DB_CREDENTIAL_MAP)


class BehaviorProjectLimsApi(BehaviorProjectBase):
    def __init__(self, lims_engine, mtrain_engine, app_engine):
        """ Downloads visual behavior data from the Allen Institute's
        internal Laboratory Information Management System (LIMS). Only
        functional if connected to the Allen Institute Network. Used to load
        data into BehaviorProjectCache.

        Typically want to construct an instance of this class by calling
            `BehaviorProjectLimsApi.default()`.

        Set log level to debug to see SQL queries dumped by
        "BehaviorProjectLimsApi" logger.

        Note -- Currently the app engine is unused because we aren't yet
        supporting the download of stimulus templates for visual behavior
        data. This feature will be added at a later date.

        Parameters
        ----------
        lims_engine :
            used for making queries against the LIMS postgres database. Must
            implement:
                select : takes a postgres query as a string. Returns a pandas
                    dataframe of results
                fetchall : takes a postgres query as a string. If there is
                    exactly one column in the response, return the values as a
                    list.
        mtrain_engine :
            used for making queries against the mtrain postgres database. Must
            implement:
                select : takes a postgres query as a string. Returns a pandas
                    dataframe of results
                fetchall : takes a postgres query as a string. If there is
                    exactly one column in the response, return the values as a
                    list.
        app_engine :
            used for making queries agains the lims web application. Must
            implement:
                stream : takes a url as a string. Returns an iterable yielding
                the response body as bytes.
        """
        self.lims_engine = lims_engine
        self.mtrain_engine = mtrain_engine
        self.app_engine = app_engine
        self.logger = logging.getLogger("BehaviorProjectLimsApi")

    @classmethod
    def default(
        cls,
        lims_credentials: Optional[DbCredentials] = None,
        mtrain_credentials: Optional[DbCredentials] = None,
        app_kwargs: Optional[Dict[str, Any]] = None) -> \
            "BehaviorProjectLimsApi":
        """Construct a BehaviorProjectLimsApi instance with default
        postgres and app engines.

        Parameters
        ----------
        lims_credentials: Optional[DbCredentials]
            Credentials to pass to the postgres connector to the lims database.
            If left unspecified, will check environment variables for the
            appropriate values.
        mtrain_credentials: Optional[DbCredentials]
            Credentials to pass to the postgres connector to the mtrain
            database. If left unspecified, will check environment variables
            for the appropriate values.
        app_kwargs: Dict
            Dict of arguments to pass to the app engine. Currently unused.

        Returns
        -------
        BehaviorProjectLimsApi
        """

        _app_kwargs = {"scheme": "http", "host": "lims2"}
        if app_kwargs:
            _app_kwargs.update(app_kwargs)
        if lims_credentials:
            lims_engine = PostgresQueryMixin(
                dbname=lims_credentials.dbname, user=lims_credentials.user,
                host=lims_credentials.host, password=lims_credentials.password,
                port=lims_credentials.port)
        else:
            # Currying is equivalent to decorator syntactic sugar
            lims_engine = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
                           (PostgresQueryMixin)())

        if mtrain_credentials:
            mtrain_engine = PostgresQueryMixin(
                dbname=lims_credentials.dbname, user=lims_credentials.user,
                host=lims_credentials.host, password=lims_credentials.password,
                port=lims_credentials.port)
        else:
            # Currying is equivalent to decorator syntactic sugar
            mtrain_engine = (
                credential_injector(MTRAIN_DB_CREDENTIAL_MAP)
                (PostgresQueryMixin)())

        app_engine = HttpEngine(**_app_kwargs)
        return cls(lims_engine, mtrain_engine, app_engine)

    @staticmethod
    def _build_in_list_selector_query(
            col,
            valid_list: Optional[SupportsStr] = None,
            operator: str = "WHERE") -> str:
        """
        Filter for rows where the value of a column is contained in a list.
        If no list is specified in `valid_list`, return an empty string.

        NOTE: if string ids are used, then the strings in `valid_list` must
        be enclosed in single quotes, or else the query will throw a column
        does not exist error. E.g. ["'mystringid1'", "'mystringid2'"...]

        :param col: name of column to compare if in a list
        :type col: str
        :param valid_list: iterable of values that can be mapped to str
            (e.g. string, int, float).
        :type valid_list: list
        :param operator: SQL operator to start the clause. Default="WHERE".
            Valid inputs: "AND", "OR", "WHERE" (not case-sensitive).
        :type operator: str
        """
        if not valid_list:
            return ""
        session_query = (
            f"""{operator} {col} IN ({",".join(
                sorted(set(map(str, valid_list))))})""")
        return session_query

    @staticmethod
    def _build_experiment_from_session_query() -> str:
        """Aggregate sql sub-query to get all ophys_experiment_ids associated
        with a single ophys_session_id."""
        query = f"""
            -- -- begin getting all ophys_experiment_ids -- --
            SELECT
                (ARRAY_AGG(DISTINCT(oe.id))) AS experiment_ids, os.id
            FROM ophys_sessions os
            RIGHT JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            GROUP BY os.id
            -- -- end getting all ophys_experiment_ids -- --
        """
        return query

    @staticmethod
    def _build_line_from_donor_query(line="driver") -> str:
        """Sub-query to get a line from a donor.
        :param line: 'driver' or 'reporter'
        """
        query = f"""
            -- -- begin getting {line} line from donors -- --
            SELECT ARRAY_AGG (g.name) AS {line}_line, d.id AS donor_id
            FROM donors d
            LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
            LEFT JOIN genotypes g ON g.id=dg.genotype_id
            LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
            WHERE gt.name='{line}'
            GROUP BY d.id
            -- -- end getting {line} line from donors -- --
        """
        return query

    def _get_behavior_summary_table(self,
                                    session_sub_query: str) -> pd.DataFrame:
        """Build and execute query to retrieve summary data for all data,
        or a subset of session_ids (via the session_sub_query).
        Should pass an empty string to `session_sub_query` if want to get
        all data in the database.
        :param session_sub_query: additional filtering logic to get a
        subset of sessions.
        :type session_sub_query: str
        :rtype: pd.DataFrame
        """
        query = f"""
            SELECT
                bs.id AS behavior_session_id,
                bs.ophys_session_id,
                bs.behavior_training_id,
                equipment.name as equipment_name,
                bs.date_of_acquisition,
                d.id as donor_id,
                d.full_genotype,
                reporter.reporter_line,
                driver.driver_line,
                g.name AS sex,
                DATE_PART('day', bs.date_of_acquisition - d.date_of_birth)
                    AS age_in_days,
                bs.foraging_id
            FROM behavior_sessions bs
            JOIN donors d on bs.donor_id = d.id
            JOIN genders g on g.id = d.gender_id
            JOIN (
                {self._build_line_from_donor_query("reporter")}
            ) reporter on reporter.donor_id = d.id
            JOIN (
                {self._build_line_from_donor_query("driver")}
            ) driver on driver.donor_id = d.id
            JOIN equipment ON equipment.id = bs.equipment_id
            {session_sub_query}
        """
        self.logger.debug(f"get_behavior_session_table query: \n{query}")
        return self.lims_engine.select(query)

    def _get_foraging_ids_from_behavior_session(
            self, behavior_session_ids: List[int]) -> List[str]:
        behav_ids = self._build_in_list_selector_query("id",
                                                       behavior_session_ids,
                                                       operator="AND")
        forag_ids_query = f"""
            SELECT foraging_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            {behav_ids};
            """
        self.logger.debug("get_foraging_ids_from_behavior_session query: \n"
                          f"{forag_ids_query}")
        foraging_ids = self.lims_engine.fetchall(forag_ids_query)

        self.logger.debug(f"Retrieved {len(foraging_ids)} foraging ids for"
                          f" behavior stage query. Ids = {foraging_ids}")
        return foraging_ids

    def _get_behavior_stage_table(
            self,
            behavior_session_ids: Optional[List[int]] = None):
        # Select fewer rows if possible via behavior_session_id
        if behavior_session_ids:
            foraging_ids = self._get_foraging_ids_from_behavior_session(
                behavior_session_ids)
            foraging_ids = [f"'{fid}'" for fid in foraging_ids]
        # Otherwise just get the full table from mtrain
        else:
            foraging_ids = None

        foraging_ids_query = self._build_in_list_selector_query(
            "bs.id", foraging_ids)

        query = f"""
            SELECT
                stages.name as session_type,
                bs.id AS foraging_id
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            {foraging_ids_query};
        """
        self.logger.debug(f"_get_behavior_stage_table query: \n {query}")
        return self.mtrain_engine.select(query)

    def get_session_data(self, ophys_session_id: int) -> BehaviorOphysSession:
        """Returns a BehaviorOphysSession object that contains methods
        to analyze a single behavior+ophys session.
        :param ophys_session_id: id that corresponds to a behavior session
        :type ophys_session_id: int
        :rtype: BehaviorOphysSession
        """
        return BehaviorOphysSession(BehaviorOphysLimsApi(ophys_session_id))

    def _get_experiment_table(
            self,
            ophys_experiment_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Helper function for easier testing.
        Return a pd.Dataframe table with all ophys_experiment_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line

        :param ophys_experiment_ids: optional list of ophys_experiment_ids
            to include
        :rtype: pd.DataFrame
        """
        if not ophys_experiment_ids:
            self.logger.warning("Getting all ophys sessions."
                                " This might take a while.")
        experiment_query = self._build_in_list_selector_query(
            "oe.id", ophys_experiment_ids)
        query = f"""
            SELECT
                oe.id as ophys_experiment_id,
                os.id as ophys_session_id,
                bs.id as behavior_session_id,
                oec.visual_behavior_experiment_container_id as container_id,
                pr.code as project_code,
                vbc.workflow_state as container_workflow_state,
                oe.workflow_state as experiment_workflow_state,
                os.name as session_name,
                os.stimulus_name as session_type,
                equipment.name as equipment_name,
                os.date_of_acquisition,
                os.isi_experiment_id,
                os.specimen_id,
                g.name as sex,
                DATE_PART('day', os.date_of_acquisition - d.date_of_birth)
                    AS age_in_days,
                d.full_genotype,
                reporter.reporter_line,
                driver.driver_line,
                id.depth as imaging_depth,
                st.acronym as targeted_structure,
                vbc.published_at
            FROM ophys_experiments_visual_behavior_experiment_containers oec
            JOIN visual_behavior_experiment_containers vbc
                ON oec.visual_behavior_experiment_container_id = vbc.id
            JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
            JOIN ophys_sessions os ON os.id = oe.ophys_session_id
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            JOIN projects pr ON pr.id = os.project_id
            JOIN donors d ON d.id = bs.donor_id
            JOIN genders g ON g.id = d.gender_id
            JOIN (
                {self._build_line_from_donor_query(line="reporter")}
            ) reporter on reporter.donor_id = d.id
            JOIN (
                {self._build_line_from_donor_query(line="driver")}
            ) driver on driver.donor_id = d.id
            LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
            JOIN structures st ON st.id = oe.targeted_structure_id
            JOIN equipment ON equipment.id = os.equipment_id
            {experiment_query};
        """
        self.logger.debug(f"get_experiment_table query: \n{query}")
        return self.lims_engine.select(query)

    def _get_session_table(
            self,
            ophys_session_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Helper function for easier testing.
        Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line

        :param ophys_session_ids: optional list of ophys_session_ids to include
        :rtype: pd.DataFrame
        """
        if not ophys_session_ids:
            self.logger.warning("Getting all ophys sessions."
                                " This might take a while.")
        session_query = self._build_in_list_selector_query("os.id",
                                                           ophys_session_ids)
        query = f"""
            SELECT
                os.id as ophys_session_id,
                bs.id as behavior_session_id,
                experiment_ids as ophys_experiment_id,
                pr.code as project_code,
                os.name as session_name,
                os.stimulus_name as session_type,
                equipment.name as equipment_name,
                os.date_of_acquisition,
                os.specimen_id,
                g.name as sex,
                DATE_PART('day', os.date_of_acquisition - d.date_of_birth)
                    AS age_in_days,
                d.full_genotype,
                reporter.reporter_line,
                driver.driver_line
            FROM ophys_sessions os
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            JOIN projects pr ON pr.id = os.project_id
            JOIN donors d ON d.id = bs.donor_id
            JOIN genders g ON g.id = d.gender_id
            JOIN (
                {self._build_experiment_from_session_query()}
            ) exp_ids ON os.id = exp_ids.id
            JOIN (
                {self._build_line_from_donor_query(line="reporter")}
            ) reporter on reporter.donor_id = d.id
            JOIN (
                {self._build_line_from_donor_query(line="driver")}
            ) driver on driver.donor_id = d.id
            JOIN equipment ON equipment.id = os.equipment_id
            {session_query};
        """
        self.logger.debug(f"get_session_table query: \n{query}")
        return self.lims_engine.select(query)

    def get_session_table(
            self,
            ophys_session_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line

        :param ophys_session_ids: optional list of ophys_session_ids to include
        :rtype: pd.DataFrame
        """
        # There is one ophys_session_id from 2018 that has multiple behavior
        # ids, causing duplicates -- drop all dupes for now; # TODO
        table = (self._get_session_table(ophys_session_ids)
                 .drop_duplicates(subset=["ophys_session_id"], keep=False)
                 .set_index("ophys_session_id"))
        return table

    def get_behavior_only_session_data(
            self, behavior_session_id: int) -> BehaviorDataSession:
        """Returns a BehaviorDataSession object that contains methods to
        analyze a single behavior session.
        :param behavior_session_id: id that corresponds to a behavior session
        :type behavior_session_id: int
        :rtype: BehaviorDataSession
        """
        return BehaviorDataSession(BehaviorDataLimsApi(behavior_session_id))

    def get_experiment_table(
            self,
            ophys_experiment_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_experiment_ids and
        relevant metadata. This is the most specific and most informative
        level to examine the data.
        Return columns:
            ophys_experiment_id, ophys_session_id, behavior_session_id,
            container_id, project_code, container_workflow_state,
            experiment_workflow_state, session_name, session_type,
            equipment_name, date_of_acquisition, isi_experiment_id,
            specimen_id, sex, age_in_days, full_genotype, reporter_line,
            driver_line, imaging_depth, targeted_structure, published_at
        :param ophys_experiment_ids: optional list of ophys_experiment_ids
            to include
        :rtype: pd.DataFrame
        """
        return self._get_experiment_table().set_index("ophys_experiment_id")

    def get_behavior_only_session_table(
            self,
            behavior_session_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.

        Can't return age at time of session because there is no field for
        acquisition date for behavior sessions (only in the stimulus pkl file)
        :rtype: pd.DataFrame
        """
        self.logger.warning("Getting behavior-only session data. "
                            "This might take a while...")
        session_query = self._build_in_list_selector_query(
            "bs.id", behavior_session_ids)
        summary_tbl = self._get_behavior_summary_table(session_query)
        stimulus_names = self._get_behavior_stage_table(behavior_session_ids)
        return (summary_tbl.merge(stimulus_names,
                                  on=["foraging_id"], how="left")
                .set_index("behavior_session_id"))

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        raise NotImplementedError()

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        raise NotImplementedError()
