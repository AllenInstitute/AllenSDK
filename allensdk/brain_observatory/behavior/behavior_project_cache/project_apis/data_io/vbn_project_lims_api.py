import pandas as pd
from allensdk.brain_observatory.behavior.\
       behavior_project_cache.project_apis.\
       data_io.behavior_project_lims_api import BehaviorProjectLimsApi


class VBNProjectLimsApi(BehaviorProjectLimsApi):

    #def _get_behavior_summary_table(self):
    #    raise NotImplementedError()

    @property
    def data_release_date(self):
        raise RuntimeError("should not be relying on data release date")

    @property
    def behavior_ophys_sessions(self):
        return []

    @property
    def behavior_only_sessions(self):
        donor_str = """(1024038404,1078586885,1038299144,1051920918,1057598487,1023232536,1000324121,1091250203,1100036636,1054702626,1088250937,1067599948,1005252690,1066191445,1134343253,1079572057,1035469403,1102157407,1099073123,1055401572,1115936367,1098595953,1049750648,1083934330,1064933502,1024938124,1053309580,1024039055,1006391440,1052679314,1075310738,1029486741,1051366038,1062711964,1056087710,1080503454,1051360928,1074838695,1091837607,1056092845,1051906227,1043723977,1071683800,1097085666,1076711655,1057575664,1052749560,1072728313,1076654843,1046926079,1087544065,1023232770,1052713734,1064158477,1070663443,1026713886,1109521182,1039843634,1051905332,1097696571,1022743357,1022743363,1030967622,1023230290,1033845075,1056087380,1096936276,1091281239,1061693281,1080378213,1090574186,1095656306,1051363699,1033846133,1072729465,1060089748,1114222996,1113647000,1087519142,1051359676,1042036158,1063385030,1078585800,1103035848,1038297549,1087316944,1068696543,1052760035,1104573423,1066195455)"""

        query=f"""
        select
        distinct(behavior_sessions.id) as behavior_session_id
        from
        behavior_sessions
        join
        donors
        on donors.id = behavior_sessions.donor_id
        where
        donors.id in {donor_str}
        and
        behavior_sessions.ophys_session_id is NULL
        and
        behavior_sessions.ecephys_session_id is NULL
        and behavior_sessions.date_of_acquisition is not NULL"""

        result = self.lims_engine.select(query)
        return result.behavior_session_id.tolist()

    @property
    def behavior_ecephys_sessions(self):
        query = f"""
        SELECT behavior_sessions.id as behavior_session_id
        FROM behavior_sessions
        JOIN ecephys_sessions
        ON behavior_sessions.ecephys_session_id=ecephys_sessions.id
        WHERE ecephys_sessions.id in {self.ecephys_sessions}"""
        result = self.lims_engine.select(query)
        return result.behavior_session_id.tolist()

    @property
    def behavior_sessions_query(self):
        full_list = self.behavior_only_sessions + self.behavior_ecephys_sessions
        full_set = set(full_list)
        full_list = list(full_set)
        full_list.sort()
        result = """("""
        for session_id in full_list:
            result += f"""{session_id},"""
        result = result[:-1]+""")"""
        return result

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
        #return new_sessions
        #return """(829720705, 755434585, 1039257177, 1041083421, 1041287144)"""
        return new_sessions


    def get_units_table(self) -> pd.DataFrame:
        query = """
        select
        eu.id as unit_id
        ,eu.ecephys_channel_id
        ,ep.id as ecephys_probe_id
        ,es.id as ecephys_session_id
        ,eu.snr
        ,eu.firing_rate
        ,eu.isi_violations
        ,eu.presence_ratio
        ,eu.amplitude_cutoff
        ,eu.isolation_distance
        ,eu.l_ratio
        ,eu.d_prime
        ,eu.nn_hit_rate
        ,eu.nn_miss_rate
        ,eu.silhouette_score
        ,eu.max_drift
        ,eu.cumulative_drift
        ,eu.duration as waveform_duration
        ,eu.halfwidth as waveform_halfwidth
        ,eu.\"PT_ratio\" as waveform_PT_ratio
        ,eu.repolarization_slope as waveform_repolarization_slope
        ,eu.recovery_slope as waveform_recovery_slope
        ,eu.amplitude as waveform_amplitude
        ,eu.spread as waveform_spread
        ,eu.velocity_above as waveform_velocity_above
        ,eu.velocity_below as waveform_velocity_below
        ,eu.local_index
        ,ec.probe_vertical_position
        ,ec.probe_horizontal_position
        ,ec.anterior_posterior_ccf_coordinate
        ,ec.dorsal_ventral_ccf_coordinate
        ,ec.manual_structure_id as ecephys_structure_id
        ,st.acronym as ecephys_structure_acronym
        """

        query += """
        FROM ecephys_units as eu
        JOIN ecephys_channels as ec on ec.id = eu.ecephys_channel_id
        JOIN ecephys_probes as ep on ep.id = ec.ecephys_probe_id
        JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
        LEFT JOIN structures as st on st.id = ec.manual_structure_id
        """

        query += f"""
        WHERE es.id IN {self.ecephys_sessions}
        """
        return self.lims_engine.select(query)

    def get_probes_table(self) -> pd.DataFrame:
        query = """
        select
        ep.id as ecephys_probe_id
        ,ep.ecephys_session_id
        ,ep.name
        ,ep.global_probe_sampling_rate as sampling_rate
        ,ep.global_probe_lfp_sampling_rate as lfp_sampling_rate
        ,ep.phase
        ,ep.use_lfp_data as has_lfp_data
        ,count(distinct(eu.id)) as unit_count
        ,count(distinct(ec.id)) as channel_count
        ,array_agg(distinct(st.acronym)) as ecephys_structure_acronyms"""

        query += """
        FROM  ecephys_probes as ep
        JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
        JOIN ecephys_channels as ec on ec.ecephys_probe_id = ep.id
        JOIN ecephys_units as eu on eu.ecephys_channel_id=ec.id
        LEFT JOIN structures st on st.id = ec.manual_structure_id"""

        query += f"""
        WHERE es.id in {self.ecephys_sessions}"""

        query += """group by ep.id"""
        return self.lims_engine.select(query)

    def get_channels_table(self) -> pd.DataFrame:
        query = """
        select
        ec.id as ecephys_channel_id
        ,ec.ecephys_probe_id
        ,es.id as ecephys_session_id
        ,ec.local_index
        ,ec.probe_vertical_position
        ,ec.probe_horizontal_position
        ,ec.anterior_posterior_ccf_coordinate
        ,ec.dorsal_ventral_ccf_coordinate
        ,ec.left_right_ccf_coordinate
        ,st.acronym as ecephys_structure_acronym
        ,count(distinct(eu.id)) as unit_count
        """

        query += """
        FROM  ecephys_channels as ec
        JOIN ecephys_probes as ep on ec.ecephys_probe_id = ep.id
        JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
        JOIN ecephys_units as eu on eu.ecephys_channel_id=ec.id
        LEFT JOIN structures st on st.id = ec.manual_structure_id"""

        query += f"""
        WHERE es.id in {self.ecephys_sessions}"""

        query += """group by ec.id, es.id, st.acronym"""
        return self.lims_engine.select(query)

    def get_vbn_behavior_only_session_table(self) -> pd.DataFrame:
        query = """
            SELECT
            coalesce(es.id, -999) AS ecephys_session_id
            ,bs.id as behavior_session_id
            ,coalesce(bs.date_of_acquisition, es.date_of_acquisition) as date_of_acquisition
            ,equipment.name as equipment_name
            ,es.stimulus_name as session_type
            ,d.full_genotype as genotype
            ,d.external_donor_name AS mouse_id
            ,g.name AS sex
            ,pr.code as project_code
            ,DATE_PART('day', coalesce(bs.date_of_acquisition, es.date_of_acquisition) - d.date_of_birth)
                  AS age_in_days
            """

        query += f"""
            FROM behavior_sessions as bs
            JOIN donors d on bs.donor_id = d.id
            JOIN genders g on g.id = d.gender_id
            LEFT OUTER JOIN equipment on equipment.id = bs.equipment_id
            LEFT OUTER JOIN ecephys_sessions es on bs.ecephys_session_id = es.id
            LEFT OUTER JOIN projects pr on pr.id = es.project_id
            WHERE bs.id in {self.behavior_sessions_query}"""

        self.logger.debug(f"get_behavior_session_table query: \n{query}")
        return self.lims_engine.select(query)

    def _get_ecephys_summary_table(self) -> pd.DataFrame:
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
            ,d.external_donor_name as mouse_id
            ,d.full_genotype as genotype
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

    def _get_counts_per_session(self) -> pd.DataFrame:
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

    def _get_structure_acronyms(self) -> pd.DataFrame:
        query = f"""
        SELECT ecephys_sessions.id as ecephys_session_id,
        array_agg(distinct(structures.acronym)) as ecephys_structure_acronyms
        FROM ecephys_sessions
        JOIN ecephys_probes
        ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
        JOIN ecephys_channels
        ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
        LEFT JOIN structures
        ON structures.id = ecephys_channels.manual_structure_id
        WHERE ecephys_sessions.id in {self.ecephys_sessions}
        GROUP BY ecephys_sessions.id"""
        return self.lims_engine.select(query)

    def get_ecephys_session_table(self) -> pd.DataFrame:
        """Returns a pd.DataFrame table with all behavior session_ids to the
        user with additional metadata.

        Can't return age at time of session because there is no field for
        acquisition date for behavior sessions (only in the stimulus pkl file)
        :rtype: pd.DataFrame
        """
        summary_tbl = self._get_ecephys_summary_table()
        ct_tbl = self._get_counts_per_session()
        summary_tbl = summary_tbl.join(
                               ct_tbl.set_index(self.index_column_name),
                               on=self.index_column_name,
                               how='left')

        struct_tbl = self._get_structure_acronyms()
        summary_tbl = summary_tbl.join(
                         struct_tbl.set_index(self.index_column_name),
                         on=self.index_column_name,
                         how='left')

        return summary_tbl

