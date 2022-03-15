import pandas as pd
from allensdk.brain_observatory.behavior.\
       behavior_project_cache.project_apis.\
       data_io.behavior_project_lims_api import BehaviorProjectLimsApi


class VBNProjectLimsApi(BehaviorProjectLimsApi):

    @property
    def data_release_date(self):
        raise RuntimeError("should not be relying on data release date")

    @property
    def index_column_name(self):
        return "ecephys_session_id"

    @property
    def ecephys_sessions(self):
        return """(829720705, 755434585)"""


    def _get_behavior_summary_table(self) -> pd.DataFrame:
        """Build and execute query to retrieve summary data for all data,
        or a subset of session_ids (via the session_sub_query).
        Should pass an empty string to `session_sub_query` if want to get
        all data in the database.
        :rtype: pd.DataFrame
        """
        query = """
            SELECT
            es.id AS ecephys_session_id
            ,es.date_of_acquisition
            ,equipment.name as equipment_name
            ,d.id as donor_id
            ,d.full_genotype
            ,d.external_donor_name AS mouse_id
            ,g.name AS sex
            ,DATE_PART('day', es.date_of_acquisition - d.date_of_birth)
                  AS age_in_days
            ,es.foraging_id
            """

        query += f"""
            FROM ecephys_sessions as es
            JOIN specimens s on s.id = es.specimen_id
            JOIN donors d on s.donor_id = d.id
            JOIN genders g on g.id = d.gender_id
            LEFT OUTER JOIN equipment on equipment.id = es.equipment_id
            WHERE es.id in {self.ecephys_sessions}"""

        self.logger.debug(f"get_behavior_session_table query: \n{query}")
        return self.lims_engine.select(query)
