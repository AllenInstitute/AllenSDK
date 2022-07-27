import argschema
import pathlib
import time
import pandas as pd

import allensdk

from allensdk.core.dataframe_utils import (
    patch_df_from_other)

from allensdk.brain_observatory.vbn_2022.metadata_writer.schemas import (
    VBN2022MetadataWriterInputSchema,
    DataReleaseToolsInputSchema)

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.id_generator import (
        FileIDGenerator)

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.utils import (
        add_file_paths_to_metadata_table)

from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    get_list_of_bad_probe_ids,
    units_table_from_ecephys_session_id_list,
    probes_table_from_ecephys_session_id_list,
    channels_table_from_ecephys_session_id_list,
    session_tables_from_ecephys_session_id_list)

from allensdk.brain_observatory.vbn_2022.\
    metadata_writer.dataframe_manipulations import (
        strip_substructure_acronym_df)

from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP,
    MTRAIN_DB_CREDENTIAL_MAP)

from allensdk.internal.api import db_connection_creator


class VBN2022MetadataWriterClass(argschema.ArgSchemaParser):
    default_schema = VBN2022MetadataWriterInputSchema
    default_output_schema = DataReleaseToolsInputSchema

    def write_df(
            self,
            df: pd.DataFrame,
            output_path: str) -> None:
        """
        Write a dataframe to the specified path as a csv,
        recording the path in self.files_written
        """
        df.to_csv(output_path, index=False)
        self.files_written.append(output_path)
        self.logger.info(f"Wrote {output_path} after "
                         f"{time.time()-self.t0: .2e} seconds")

    def run(self):
        self.t0 = time.time()

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
        self.files_written = []

        units_table = units_table_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        units_table = strip_substructure_acronym_df(
                df=units_table,
                col_name='structure_acronym')

        units_table = units_table[
            ["unit_id",
             "ecephys_channel_id",
             "ecephys_probe_id",
             "ecephys_session_id",
             "amplitude_cutoff",
             "anterior_posterior_ccf_coordinate",
             "dorsal_ventral_ccf_coordinate",
             "left_right_ccf_coordinate",
             "cumulative_drift",
             "d_prime",
             "structure_acronym",
             "structure_id",
             "firing_rate",
             "isi_violations",
             "isolation_distance",
             "l_ratio",
             "local_index",
             "max_drift",
             "nn_hit_rate",
             "nn_miss_rate",
             "presence_ratio",
             "probe_horizontal_position",
             "probe_vertical_position",
             "silhouette_score",
             "snr",
             "valid_data",
             "amplitude",
             "waveform_duration",
             "waveform_halfwidth",
             "PT_ratio",
             "recovery_slope",
             "repolarization_slope",
             "spread",
             "velocity_above",
             "velocity_below"]]

        self.write_df(
            df=units_table,
            output_path=self.args['units_path'])

        probes_table = probes_table_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        probes_table = strip_substructure_acronym_df(
                df=probes_table,
                col_name='structure_acronyms')

        probes_table.drop(
            labels=['temporal_subsampling_factor'],
            axis='columns',
            inplace=True)

        self.write_df(
            df=probes_table,
            output_path=self.args['probes_path'])

        channels_table = channels_table_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    ecephys_session_id_list=session_id_list,
                    probe_ids_to_skip=probe_ids_to_skip)

        channels_table = strip_substructure_acronym_df(
                df=channels_table,
                col_name='structure_acronym')

        channels_table.drop(
                    labels=['structure_id'],
                    axis='columns',
                    inplace=True)

        self.write_df(
            df=channels_table,
            output_path=self.args['channels_path'])

        failed_session_list = self.args[
            'failed_ecephys_session_id_list']

        (ecephys_session_table,
         behavior_session_table) = session_tables_from_ecephys_session_id_list(
                    lims_connection=lims_connection,
                    mtrain_connection=mtrain_connection,
                    ecephys_session_id_list=session_id_list,
                    failed_ecephys_session_id_list=failed_session_list,
                    probe_ids_to_skip=probe_ids_to_skip,
                    logger=self.logger)

        ecephys_nwb_dir = pathlib.Path(
                                self.args['ecephys_nwb_dir'])

        ecephys_session_table = add_file_paths_to_metadata_table(
                    metadata_table=ecephys_session_table,
                    id_generator=file_id_generator,
                    file_dir=ecephys_nwb_dir,
                    file_prefix=self.args['ecephys_nwb_prefix'],
                    index_col='ecephys_session_id',
                    on_missing_file=self.args['on_missing_file'])

        # add supplemental columns to the ecephys_sessions
        # column
        if self.args['supplemental_data'] is not None:
            self.logger.info("Adding supplemental data")
            supplemental_df = pd.DataFrame(
                    data=self.args['supplemental_data'])

            columns_to_patch = []
            for column_name in supplemental_df.columns:
                if column_name == 'ecephys_session_id':
                    continue
                columns_to_patch.append(column_name)

            ecephys_session_table = patch_df_from_other(
                    target_df=ecephys_session_table,
                    source_df=supplemental_df,
                    columns_to_patch=columns_to_patch,
                    index_column='ecephys_session_id')

        self.write_df(
            df=ecephys_session_table,
            output_path=self.args['ecephys_sessions_path'])

        self.write_df(
            df=behavior_session_table,
            output_path=self.args['behavior_sessions_path'])

        pipeline_metadata = []
        sdk_metadata = {"name": "AllenSDK",
                        "version": str(allensdk.__version__),
                        "comment": ""}
        pipeline_metadata.append(sdk_metadata)

        output_data = {
            'metadata_files': self.files_written,
            'data_pipeline_metadata': pipeline_metadata,
            'project_name': 'visual-behavior-neuropixels',
            'log_level': 'INFO'
        }

        self.output(output_data, indent=2)

        self.logger.info(f"Wrote {self.args['ecephys_sessions_path']}\n"
                         f"and {self.args['behavior_sessions_path']}\n"
                         f"after {time.time()-self.t0:.2e} seconds")
