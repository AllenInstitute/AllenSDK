import argschema

from allensdk.brain_observatory.vbn_2022.metadata_writer.schemas import (
    VBN2022MetadataWriterInputSchema)

from allensdk.brain_observatory.vbn_2022.metadata_writer.utils import (
    _get_units_table,
    _get_probes_table,
    _get_channels_table,
    _get_ecephys_session_table)

from allensdk.brain_observatory.vbn_2022.metadata_writer.session_utils import (
    _postprocess_sessions)

from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator


class VBN2022MetadataWriterClass(argschema.ArgSchemaParser):
    default_schema = VBN2022MetadataWriterInputSchema

    def run(self):
        lims_connection = db_connection_creator(
                fallback_credentials=LIMS_DB_CREDENTIAL_MAP
            )

        session_id_list = self.args['ecephys_session_id_list']

        units_table = _get_units_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list)

        probes_table = _get_probes_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list)

        channels_table = _get_channels_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list)

        session_table = _get_ecephys_session_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list)

        session_table = _postprocess_sessions(
                            sessions_df=session_table)

        print(session_table)
        print(session_table.columns)
