import pandas as pd
from typing import Optional, List, Dict, Any, Iterable
import logging

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.abcs import BehaviorProjectBase  # noqa: E501
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi, BehaviorOphysLimsApi)
from allensdk.internal.api import db_connection_creator
from allensdk.brain_observatory.ecephys.ecephys_project_api.http_engine \
    import (HttpEngine)
from allensdk.core.typing import SupportsStr
from allensdk.core.authentication import DbCredentials
from allensdk.core.auth_config import (
    MTRAIN_DB_CREDENTIAL_MAP, LIMS_DB_CREDENTIAL_MAP)


class BehaviorProjectLimsApi(BehaviorProjectBase):
    def __init__(self, lims_engine, mtrain_engine, app_engine,
                 data_release_date: Optional[str] = None):
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
        data_release_date
            Use to filter tables to only include data released on date
            ie 2021-03-25
        """
        self.lims_engine = lims_engine
        self.mtrain_engine = mtrain_engine
        self.app_engine = app_engine
        self.data_release_date = data_release_date
        self.logger = logging.getLogger("BehaviorProjectLimsApi")

    @classmethod
    def default(
            cls,
            lims_credentials: Optional[DbCredentials] = None,
            mtrain_credentials: Optional[DbCredentials] = None,
            app_kwargs: Optional[Dict[str, Any]] = None,
            data_release_date: Optional[str] = None) -> \
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
        data_release_date: Optional[str]
            Filters tables to include only data released on date
            ie 2021-03-25
        app_kwargs: Dict
            Dict of arguments to pass to the app engine. Currently unused.

        Returns
        -------
        BehaviorProjectLimsApi
        """

        _app_kwargs = {"scheme": "http", "host": "lims2"}
        if app_kwargs:
            _app_kwargs.update(app_kwargs)

        lims_engine = db_connection_creator(
            credentials=lims_credentials,
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)
        mtrain_engine = db_connection_creator(
            credentials=mtrain_credentials,
            fallback_credentials=MTRAIN_DB_CREDENTIAL_MAP)

        app_engine = HttpEngine(**_app_kwargs)
        return cls(lims_engine, mtrain_engine, app_engine,
                   data_release_date=data_release_date)

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

    def _build_experiment_from_session_query(self) -> str:
        """Aggregate sql sub-query to get all ophys_experiment_ids associated
        with a single ophys_session_id."""
        if self.data_release_date:
            release_filter = self._get_ophys_experiment_release_filter()
        else:
            release_filter = ''
        query = f"""
            -- -- begin getting all ophys_experiment_ids -- --
            SELECT
                (ARRAY_AGG(DISTINCT(oe.id))) AS experiment_ids, os.id
            FROM ophys_sessions os
            RIGHT JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            {release_filter}
            GROUP BY os.id
            -- -- end getting all ophys_experiment_ids -- --
        """
        return query

    def _build_container_from_session_query(self) -> str:
        """Aggregate sql sub-query to get all ophys_container_ids associated
        with a single ophys_session_id."""
        if self.data_release_date:
            release_filter = self._get_ophys_experiment_release_filter()
        else:
            release_filter = ''
        query = f"""
            -- -- begin getting all ophys_container_ids -- --
            SELECT
                (ARRAY_AGG(
                        DISTINCT(oec.visual_behavior_experiment_container_id))
                    ) AS container_ids, os.id
            FROM ophys_experiments_visual_behavior_experiment_containers oec
            JOIN visual_behavior_experiment_containers vbc
                ON oec.visual_behavior_experiment_container_id = vbc.id
            JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
            JOIN ophys_sessions os ON os.id = oe.ophys_session_id
            {release_filter}
            GROUP BY os.id
            -- -- end getting all ophys_container_ids -- --
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

    def _get_behavior_summary_table(self) -> pd.DataFrame:
        """Build and execute query to retrieve summary data for all data,
        or a subset of session_ids (via the session_sub_query).
        Should pass an empty string to `session_sub_query` if want to get
        all data in the database.
        :rtype: pd.DataFrame
        """
        query = f"""
            SELECT
                bs.id AS behavior_session_id,
                equipment.name as equipment_name,
                bs.date_of_acquisition,
                d.id as donor_id,
                d.full_genotype,
                d.external_donor_name AS mouse_id,
                reporter.reporter_line,
                driver.driver_line,
                g.name AS sex,
                DATE_PART('day', bs.date_of_acquisition - d.date_of_birth)
                    AS age_in_days,
                bs.foraging_id
            FROM behavior_sessions bs
            JOIN donors d on bs.donor_id = d.id
            JOIN genders g on g.id = d.gender_id
            LEFT OUTER JOIN (
                {self._build_line_from_donor_query("reporter")}
            ) reporter on reporter.donor_id = d.id
            LEFT OUTER JOIN (
                {self._build_line_from_donor_query("driver")}
            ) driver on driver.donor_id = d.id
            LEFT OUTER JOIN equipment ON equipment.id = bs.equipment_id
        """

        if self.data_release_date is not None:
            query += self._get_behavior_session_release_filter()

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

    def get_behavior_stage_parameters(self,
                                      foraging_ids: List[str]) -> pd.Series:
        """Gets the stage parameters for each foraging id from mtrain

        Parameters
        ----------
        foraging_ids
            List of foraging ids


        Returns
        ---------
        Series with index of foraging id and values stage parameters
        """
        foraging_ids_query = self._build_in_list_selector_query(
            "bs.id", foraging_ids)

        query = f"""
            SELECT
                bs.id AS foraging_id,
                stages.parameters as stage_parameters
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            {foraging_ids_query};
        """
        df = self.mtrain_engine.select(query)
        df = df.set_index('foraging_id')
        return df['stage_parameters']

    def get_behavior_ophys_experiment(self, ophys_experiment_id: int
                                      ) -> BehaviorOphysExperiment:
        """Returns a BehaviorOphysExperiment object that contains methods
        to analyze a single behavior+ophys session.
        :param ophys_experiment_id: id that corresponds to an ophys experiment
        :type ophys_experiment_id: int
        :rtype: BehaviorOphysExperiment
        """
        return BehaviorOphysExperiment(
                BehaviorOphysLimsApi(ophys_experiment_id))

    def _get_ophys_experiment_table(self) -> pd.DataFrame:
        """
        Helper function for easier testing.
        Return a pd.Dataframe table with all ophys_experiment_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line, mouse_id

        :rtype: pd.DataFrame
        """
        query = """
            SELECT
                oe.id as ophys_experiment_id,
                os.id as ophys_session_id,
                os.stimulus_name as session_type,
                bs.id as behavior_session_id,
                oec.visual_behavior_experiment_container_id as
                    ophys_container_id,
                pr.code as project_code,
                vbc.workflow_state as container_workflow_state,
                oe.workflow_state as experiment_workflow_state,
                os.name as session_name,
                os.date_of_acquisition,
                os.isi_experiment_id,
                id.depth as imaging_depth,
                st.acronym as targeted_structure,
                vbc.published_at
            FROM ophys_experiments_visual_behavior_experiment_containers oec
            JOIN visual_behavior_experiment_containers vbc
                ON oec.visual_behavior_experiment_container_id = vbc.id
            JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
            JOIN ophys_sessions os ON os.id = oe.ophys_session_id
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            LEFT OUTER JOIN projects pr ON pr.id = os.project_id
            LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
            JOIN structures st ON st.id = oe.targeted_structure_id
        """

        if self.data_release_date is not None:
            query += self._get_ophys_experiment_release_filter()

        self.logger.debug(f"get_ophys_experiment_table query: \n{query}")
        return self.lims_engine.select(query)

    def _get_ophys_session_table(self) -> pd.DataFrame:
        """Helper function for easier testing.
        Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line, mouse_id

        :rtype: pd.DataFrame
        """
        query = f"""
            SELECT
                os.id as ophys_session_id,
                bs.id as behavior_session_id,
                exp_ids.experiment_ids as ophys_experiment_id,
                cntr_ids.container_ids as ophys_container_id,
                pr.code as project_code,
                os.name as session_name,
                os.date_of_acquisition,
                os.specimen_id,
                os.stimulus_name as session_type
            FROM ophys_sessions os
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            LEFT OUTER JOIN projects pr ON pr.id = os.project_id
            JOIN (
                {self._build_experiment_from_session_query()}
            ) exp_ids ON os.id = exp_ids.id
            JOIN (
                {self._build_container_from_session_query()}
            ) cntr_ids ON os.id = cntr_ids.id
        """

        if self.data_release_date is not None:
            query += self._get_ophys_session_release_filter()
        self.logger.debug(f"get_ophys_session_table query: \n{query}")
        return self.lims_engine.select(query)

    def get_ophys_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id,
                        ophys_experiment_id, project_code, session_name,
                        session_type, equipment_name, date_of_acquisition,
                        specimen_id, full_genotype, sex, age_in_days,
                        reporter_line, driver_line
        :rtype: pd.DataFrame
        """
        # There is one ophys_session_id from 2018 that has multiple behavior
        # ids, causing duplicates -- drop all dupes for now; # TODO
        table = (self._get_ophys_session_table()
                 .drop_duplicates(subset=["ophys_session_id"], keep=False)
                 .set_index("ophys_session_id"))
        return table

    def get_behavior_session(
            self, behavior_session_id: int) -> BehaviorSession:
        """Returns a BehaviorSession object that contains methods to
        analyze a single behavior session.
        :param behavior_session_id: id that corresponds to a behavior session
        :type behavior_session_id: int
        :rtype: BehaviorSession
        """
        return BehaviorSession(BehaviorLimsApi(behavior_session_id))

    def get_ophys_experiment_table(
            self,
            ophys_experiment_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_experiment_ids and
        relevant metadata. This is the most specific and most informative
        level to examine the data.
        Return columns:
            ophys_experiment_id, ophys_session_id, behavior_session_id,
            ophys_container_id, project_code, container_workflow_state,
            experiment_workflow_state, session_name, session_type,
            equipment_name, date_of_acquisition, isi_experiment_id,
            specimen_id, sex, age_in_days, full_genotype, reporter_line,
            driver_line, imaging_depth, targeted_structure, published_at
        :param ophys_experiment_ids: optional list of ophys_experiment_ids
            to include
        :rtype: pd.DataFrame
        """
        df = self._get_ophys_experiment_table()
        return df.set_index("ophys_experiment_id")

    def get_behavior_session_table(self) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.

        Can't return age at time of session because there is no field for
        acquisition date for behavior sessions (only in the stimulus pkl file)
        :rtype: pd.DataFrame
        """
        summary_tbl = self._get_behavior_summary_table()
        stimulus_names = self._get_behavior_stage_table(
            behavior_session_ids=summary_tbl.index.tolist())
        return (summary_tbl.merge(stimulus_names,
                                  on=["foraging_id"], how="left")
                .set_index("behavior_session_id"))

    def get_release_files(self, file_type='BehaviorNwb') -> pd.DataFrame:
        """Gets the release nwb files.

        Parameters
        ----------
        file_type
            NWB files to return ('BehaviorNwb', 'BehaviorOphysNwb')

        Returns
        ---------
        Dataframe of release files and file metadata
            -index of behavior_session_id or ophys_experiment_id
            -columns file_id and isilon filepath
        """
        if self.data_release_date is None:
            raise RuntimeError('data_release_date must be set in constructor')

        if file_type not in ('BehaviorNwb', 'BehaviorOphysNwb'):
            raise ValueError(f'cannot retrieve file type {file_type}')

        if file_type == 'BehaviorNwb':
            attachable_id_alias = 'behavior_session_id'
            select_clause = f'''
                SELECT attachable_id as {attachable_id_alias}, id as file_id,
                    filename, storage_directory
            '''
            join_clause = ''
        else:
            attachable_id_alias = 'ophys_experiment_id'
            select_clause = f'''
                SELECT attachable_id as {attachable_id_alias},
                    bs.id as behavior_session_id, wkf.id as file_id,
                    filename, wkf.storage_directory
            '''
            join_clause = """
                JOIN ophys_experiments oe ON oe.id = attachable_id
                JOIN ophys_sessions os ON os.id = oe.ophys_session_id
                JOIN behavior_sessions bs on bs.ophys_session_id = os.id
            """

        query = f'''
            {select_clause}
            FROM well_known_files wkf
            {join_clause}
            WHERE published_at = '{self.data_release_date}' AND
                well_known_file_type_id IN (
                    SELECT id
                    FROM well_known_file_types
                    WHERE name = '{file_type}'
                );
        '''

        res = self.lims_engine.select(query)
        res['isilon_filepath'] = res['storage_directory'] \
            .str.cat(res['filename'])
        res = res.drop(['filename', 'storage_directory'], axis=1)
        return res.set_index(attachable_id_alias)

    def _get_behavior_session_release_filter(self):
        # 1) Get release behavior only session ids
        behavior_only_release_files = self.get_release_files(
            file_type='BehaviorNwb')
        release_behavior_only_session_ids = \
            behavior_only_release_files.index.tolist()

        # 2) Get release behavior with ophys session ids
        ophys_release_files = self.get_release_files(
            file_type='BehaviorOphysNwb')
        release_behavior_with_ophys_session_ids = \
            ophys_release_files['behavior_session_id'].tolist()

        # 3) release behavior session ids is combination
        release_behavior_session_ids = \
            release_behavior_only_session_ids + \
            release_behavior_with_ophys_session_ids

        return self._build_in_list_selector_query(
            "bs.id", release_behavior_session_ids)

    def _get_ophys_session_release_filter(self):
        release_files = self.get_release_files(
            file_type='BehaviorOphysNwb')
        return self._build_in_list_selector_query(
            "bs.id", release_files['behavior_session_id'].tolist())

    def _get_ophys_experiment_release_filter(self):
        release_files = self.get_release_files(
            file_type='BehaviorOphysNwb')
        return self._build_in_list_selector_query(
            "oe.id", release_files.index.tolist())

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        raise NotImplementedError()

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        raise NotImplementedError()
