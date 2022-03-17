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
        new_sessions = """
(1041083421,1041287144,1050962145,1051155866,
1044385384,1044594870,1056495334,1056720277,1055221968,
1055403683,1055240613,1055415082,1062755779,1063010385,
1062755416,1063010496,1070961372,1071301976,1067588044,
1067781390,1069192277,1069458330,1065449881,1065908084,
1070969560,1071300149,1077712208,1077891954,1084427055,
1084938101,1079018622,1079278078,1081079981,1081429294,
1081090969,1081431006,1099598937,1099869737,1104052767,
1104297538,1109680280,1109889304,1108335514,1108528422,
1116941914,1117148442,1118324999,1118512505,1119946360,
1120251466,1130113579,1130349290,1132595046,1132752601,
1131502356,1131648544,1128520325,1128719842,1139846596,
1140102579,1047969464,1048189115,1039257177,1039557143,
1047977240,1048196054,1052331749,1052530003,1046166369,
1046581736,1053718935,1053941483,1065437523,1065905010,
1064400234,1064644573,1069193611,1069461581,1077711823,
1077897245,1084428217,1084939136,1086200042,1086410738,
1087720624,1087992708,1089296550,1089572777,1086198651,
1086433081,1095142367,1095353694,1095138995,1095340643,
1093638203,1093867806,1092284153,1092468497,1098119201,
1098350754,1101263832,1101467873,1106985031,1107172157,
1104058216,1104289498,1101268690,1101473342,1106984390,
1107177016,1102575448,1102790314,1105543760,1105798776,
1115077618,1115356973,1108334384,1108531612,1122903357,
1123100019,1112302803,1112515874,1115086689,1115368723,
1121406444,1121607504,1118327332,1118508667,1124285719,
1124507277,1125713722,1125937457,1043752325,1044016459,
1044389060,1044597824,1049273528,1049514117,1052342277,
1052533639,1053709239,1053925378,1059678195,1059908979,
1061238668,1061463555,1064415305,1064639378,1072345110,
1072572100,1076265417,1076487758,1072341440,1072567062,
1079018673,1079275221,1067593545,1067790400,1082987883,
1083212839,1093642839,1093864136,1090803859,1091039376,
1099596266,1099872628,1087723305,1087993643,1090800639,
1091039902,1096620314,1096935816,1092283837,1092466205,
1113751921,1113957627,1111013640,1111216934,1152632711,
1152811536)
"""
        return new_sessions
        return """(829720705, 755434585, 1039257177)"""



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
            ,bs.id as behavior_session_id
            ,es.date_of_acquisition
            ,equipment.name as equipment_name
            ,es.stimulus_name as session_type
            ,d.id as donor_id
            ,d.full_genotype
            ,d.external_donor_name AS mouse_id
            ,g.name AS sex
            ,pr.code as project_code
            ,DATE_PART('day', es.date_of_acquisition - d.date_of_birth)
                  AS age_in_days
            """

        query += f"""
            FROM ecephys_sessions as es
            JOIN specimens s on s.id = es.specimen_id
            JOIN donors d on s.donor_id = d.id
            JOIN genders g on g.id = d.gender_id
            JOIN projects pr on pr.id = es.project_id
            LEFT OUTER JOIN equipment on equipment.id = es.equipment_id
            LEFT OUTER JOIN behavior_sessions bs on bs.ecephys_session_id = es.id
            WHERE es.id in {self.ecephys_sessions}"""

        self.logger.debug(f"get_behavior_session_table query: \n{query}")
        return self.lims_engine.select(query)

    def get_counts_per_session(self) -> pd.DataFrame:
        query = f"""
        SELECT ecephys_sessions.id as ecephys_session_id,
        COUNT(DISTINCT(ecephys_units.id)) as unit_count,
        COUNT(DISTINCT(ecephys_probes.id)) as probe_count,
        COUNT(DISTINCT(ecephys_channels.id)) as channel_count
        FROM ecephys_sessions
        LEFT OUTER JOIN ecephys_probes ON
        ecephys_probes.ecephys_session_id = ecephys_sessions.id
        LEFT OUTER JOIN ecephys_channels ON
        ecephys_channels.ecephys_probe_id = ecephys_probes.id
        LEFT OUTER JOIN ecephys_units ON
        ecephys_units.ecephys_channel_id = ecephys_channels.id
        WHERE ecephys_sessions.id in {self.ecephys_sessions}
        GROUP BY ecephys_sessions.id"""
        return self.lims_engine.select(query)

    def get_behavior_session_table(self) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.

        Can't return age at time of session because there is no field for
        acquisition date for behavior sessions (only in the stimulus pkl file)
        :rtype: pd.DataFrame
        """
        summary_tbl = self._get_behavior_summary_table()
        return summary_tbl

