import argschema

from allensdk.brain_observatory.vbn_2022.metadata_writer.schemas import (
    VBN2022MetadataWriterInputSchema)

from allensdk.brain_observatory.vbn_2022.metadata_writer.utils import (
    get_list_of_bad_probe_ids,
    _get_units_table,
    _get_probes_table,
    _get_channels_table,
    _get_session_tables)

from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP,
    MTRAIN_DB_CREDENTIAL_MAP)

from allensdk.internal.api import db_connection_creator


class VBN2022MetadataWriterClass(argschema.ArgSchemaParser):
    default_schema = VBN2022MetadataWriterInputSchema

    def run(self):
        lims_connection = db_connection_creator(
                fallback_credentials=LIMS_DB_CREDENTIAL_MAP
            )

        mtrain_connection = db_connection_creator(
                fallback_credentials=MTRAIN_DB_CREDENTIAL_MAP)

        if self.args['probes_to_skip'] is not None:
            probe_ids_to_skip = get_list_of_bad_probe_ids(
                        lims_connection=lims_connection,
                        probes_to_skip=self.args['probes_to_skip'])
        else:
            probe_ids_to_skip = None

        session_id_list = self.args['ecephys_session_id_list']

        units_table = _get_units_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)
        units_table.to_csv(self.args['units_path'], index=False)

        probes_table = _get_probes_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)
        probes_table.to_csv(self.args['probes_path'], index=False)

        channels_table = _get_channels_table(
                    lims_connection=lims_connection,
                    session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)
        channels_table.to_csv(self.args['channels_path'], index=False)

        (session_table,
         behavior_session_table) = _get_session_tables(
                    lims_connection=lims_connection,
                    mtrain_connection=mtrain_connection,
                    session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)
        session_table.to_csv(self.args['ecephys_sessions_path'],
                             index=False)
        behavior_session_table.to_csv(
                             self.args['behavior_sessions_path'],
                             index=False)
