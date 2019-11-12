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
        query = f"""
            SELECT
                bs.id as behavior_session_id,
                os.id as ophys_session_id,
                bs.behavior_training_id as behavior_training_id,
                os.specimen_id,
                os.isi_experiment_id,
                os.stimulus_name as session_type,
                d.full_genotype as genotype,
                g.name as sex
            JOIN donors d on bs.donor_id = d.id
            JOIN genders g. on g.id = d.gender_id
            {session_query}
        """
        return self.postgres_engine.select(query)

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        pass

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """ Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        pass
