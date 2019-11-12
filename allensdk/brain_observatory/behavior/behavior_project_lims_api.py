import pandas as pd
from typing import Iterable, Optional, List, Union
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
from allensdk.brain_observatory.ecephys.ecephys_project_api import HttpEngine


class BehaviorProjectLimsApi(BehaviorProjectBase):
    def __init__(self, postgres_engine, app_engine):
        """ Downloads visual behavior data from the Allen Institute's
        internal Laboratory Information Management System (LIMS). Only
        functional if connected to the Allen Institute Network. Used to load
        data into BehaviorProjectCache.

        Typically want to construct an instance of this class by calling
            `BehaviorProjectLimsApi.default()`.

        Parameters
        ----------
        postgres_engine :
            used for making queries against the LIMS postgres database. Must
            implement:
                select : takes a postgres query as a string. Returns a pandas
                    dataframe of results
                select_one : takes a postgres query as a string. If there is
                    exactly one record in the response, returns that record as
                    a dict. Otherwise returns an empty dict.
        app_engine :
            used for making queries agains the lims web application. Must
            implement:
                stream : takes a url as a string. Returns an iterable yielding
                the response body as bytes.
        """
        self.postgres_engine = postgres_engine
        self.app_engine = app_engine
        self.logger = logging.getLogger("BehaviorProjectLimsApi")

    @classmethod
    def default(cls, pg_kwargs=None, app_kwargs=None):

        _pg_kwargs = {}
        if pg_kwargs is not None:
            _pg_kwargs.update(pg_kwargs)

        _app_kwargs = {"scheme": "http", "host": "lims2"}
        if app_kwargs is not None:
            _app_kwargs.update(app_kwargs)

        pg_engine = PostgresQueryMixin(**_pg_kwargs)
        app_engine = HttpEngine(**_app_kwargs)
        return cls(pg_engine, app_engine)

    def get_session_data(self, ophys_session_id: int) -> BehaviorOphysSession:
        """Returns a BehaviorOphysSession object that contains methods
        to analyze a single behavior+ophys session.
        :param ophys_session_id: id that corresponds to a behavior session
        :type ophys_session_id: int
        :rtype: BehaviorOphysSession
        """
        return BehaviorOphysSession(BehaviorOphysLimsApi(ophys_session_id))

    def get_session_table(
            self, 
            ophys_session_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Return a pd.Dataframe table with all ophys_session_ids and relevant
        metadata.
        Return columns: ophys_session_id, behavior_session_id, specimen_id,
                        ophys_experiment_ids, isi_experiment_id, session_type,
                        date_of_acquisition, genotype, sex, age_in_days
        :rtype: pd.DataFrame
        """
        session_query = self._build_id_selector_query("os.id",
                                                      ophys_session_ids)
        experiment_query = self._build_experiment_from_session_query()
        query = f"""
        SELECT
            os.id as ophys_session_id,
            bs.id as behavior_session_id,
            os.specimen_id,
            os.isi_experiment_id,
            os.stimulus_name as session_type,
            os.date_of_acquisition,
            d.full_genotype as genotype,
            g.name as sex,
            DATE_PART('day', os.date_of_acquisition - d.date_of_birth)
                AS age_in_days
        FROM ophys_sessions os
        JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
        JOIN donors d ON d.id = bs.donor_id
        JOIN genders g ON g.id = d.gender_id
        JOIN (
            {experiment_query}
        ) exp_ids ON os.id = os.id
        {session_query};
        """
        self.logger.debug(f"get_session_table query: \n{query}")
        return self.postgres_engine.select(query)

    @staticmethod
    def _build_id_selector_query(
            id_col,
            session_ids: Optional[List[Union[str, int]]] = None) -> str:
        if not session_ids:
            return ""
        session_query = f"""
            WHERE {id_col} IN ({",".join(set(map(str, session_ids)))})"""
        return session_query

    @staticmethod
    def _build_experiment_from_session_query():
        query = f"""
            -- -- begin getting all ophys_experiment_ids -- --
            SELECT
                (ARRAY_AGG(DISTINCT(oe.id))) as experiment_ids, os.id
            FROM ophys_sessions os
            RIGHT JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            GROUP BY os.id
            -- -- end getting all ophys_experiment_ids -- --
        """
        return query

    def get_behavior_only_session_data(
            self, behavior_session_id: int) -> BehaviorDataSession:
        """Returns a BehaviorDataSession object that contains methods to
        analyze a single behavior session.
        :param behavior_session_id: id that corresponds to a behavior session
        :type behavior_session_id: int
        :rtype: BehaviorDataSession
        """
        return BehaviorDataSession(BehaviorDataLimsApi(behavior_session_id))

    def get_behavior_only_session_table(
            self,
            behavior_session_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.

        Can't return age at time of session because there is no field for
        acquisition date for behavior sessions (only in the stimulus pkl file)
        :rtype: pd.DataFrame
        """
        session_query = self._build_id_selector_query("bs.id",
                                                      behavior_session_ids)
        summary_tbl = self._get_behavior_summary(session_query)
        stimulus_names = self._get_behavior_stage(behavior_session_ids)
        return summary_tbl.merge(stimulus_names,
                                 on=["foraging_id"], how="left")

    def _get_behavior_summary(self, session_sub_query):
        query = f"""
            SELECT
                bs.id as behavior_session_id,
                bs.ophys_session_id,
                bs.behavior_training_id,
                sp.id as specimen_id,
                d.full_genotype as genotype,
                g.name as sex,
                bs.foraging_id
            FROM behavior_sessions bs
            JOIN donors d on bs.donor_id = d.id
            JOIN genders g. on g.id = d.gender_id
            JOIN specimens sp ON sp.donor_id = d.id
            {session_sub_query}
        """
        return self.postgres_engine.select(query)

    def _get_behavior_stage(
            self,
            behavior_session_ids: Optional[List[int]] = None,
            mtrain_db: Optional[PostgresQueryMixin] = None):
        # Select fewer rows if possible via behavior_session_id
        if behavior_session_ids:
            behav_ids = self._build_id_selector_query("id",
                                                      behavior_session_ids)            
            forag_ids_query = f"""
                SELECT foraging_id
                FROM behavior_sessions
                {behav_ids}
                AND foraging_id IS NOT NULL
                """
            foraging_ids = self.postgres_engine.fetchall(forag_ids_query)

            self.logger.debug(f"Retrieved {len(foraging_ids)} foraging ids for"
                              f" behavior stage query. Ids = {foraging_ids}")
        # Otherwise just get the full table from mtrain
        else:
            foraging_ids = None

        foraging_ids_query = self._build_id_selector_query(
            "bs.id", foraging_ids)

        # TODO: this password has already been exposed in code but we really
        # need to move towards using a secrets database
        if not mtrain_db:
            mtrain_db = PostgresQueryMixin(
                dbname="mtrain", user="mtrainreader",
                host="prodmtrain1", port=5432, password="mtrainro")
        query = f"""
            SELECT
                stages.name,
                bs.id AS foraging_id
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            {foraging_ids_query}
        """

        return mtrain_db.select(query)

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        return self._get_template(
            f"natural_movie_{number}", self.STIMULUS_TEMPLATE_NAMESPACE
        )

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        return self._get_template(
            f"natural_scene_{int(number)}", self.STIMULUS_TEMPLATE_NAMESPACE
        )

    def _get_template(self, name, namespace):
        """ Identify the WellKnownFile record associated with a stimulus
        template and stream its data if present.
        """
        query = f"""
                SELECT
                    st.well_known_file_id
                FROM stimuli st
                JOIN stimulus_namespaces sn ON sn.id = st.stimulus_namespace_id
                WHERE
                    st.name = '{name}'
                    AND sn.name = '{namespace}'
                """
        wkf_id = self.postgres_engine.fetchone(query)
        download_link = f"well_known_files/download/{wkf_id}?wkf_id={wkf_id}"
        return self.app_engine.stream(download_link)
