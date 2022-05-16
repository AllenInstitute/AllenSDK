import argschema
import pathlib

from allensdk.brain_observatory.vbn_2022.metadata_writer.schemas import (
    VBN2022MetadataWriterInputSchema)

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.id_generator import (
        FileIDGenerator)

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.utils import (
        add_file_paths_to_metadata_table)

from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    get_list_of_bad_probe_ids,
    units_table_from_ecephys_session_ids,
    probes_table_from_ecephys_session_id_list,
    channels_table_from_ecephys_session_id_list,
    session_tables_from_ecephys_session_id_list)

from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP,
    MTRAIN_DB_CREDENTIAL_MAP)

from allensdk.internal.api import db_connection_creator


class VBN2022MetadataWriterClass(argschema.ArgSchemaParser):
    default_schema = VBN2022MetadataWriterInputSchema

    def run(self):

        file_id_generator = FileIDGenerator()

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

        units_table = units_table_from_ecephys_session_ids(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)
        units_table.to_csv(self.args['units_path'], index=False)

        probes_table = probes_table_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        probes_table.drop(
            labels=['temporal_subsampling_factor'],
            axis='columns',
            inplace=True)

        probes_table.to_csv(self.args['probes_path'], index=False)

        channels_table = channels_table_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        channels_table.drop(
                    labels=['ecephys_structure_id'],
                    axis='columns',
                    inplace=True)

        channels_table.to_csv(self.args['channels_path'], index=False)

        (session_table,
         behavior_session_table) = session_tables_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    mtrain_connection=mtrain_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        ecephys_nwb_dir = pathlib.Path(
                                self.args['ecephys_nwb_dir'])

        session_table = add_file_paths_to_metadata_table(
                    metadata_table=session_table,
                    id_generator=file_id_generator,
                    file_dir=ecephys_nwb_dir,
                    file_prefix=self.args['ecephys_nwb_prefix'],
                    index_col='ecephys_session_id',
                    on_missing_file=self.args['on_missing_file'])

        session_table.to_csv(self.args['ecephys_sessions_path'],
                             index=False)
        behavior_session_table.to_csv(
                             self.args['behavior_sessions_path'],
                             index=False)
