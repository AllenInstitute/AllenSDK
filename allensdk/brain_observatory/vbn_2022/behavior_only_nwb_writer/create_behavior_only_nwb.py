import argschema
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import pynwb

from allensdk.brain_observatory.vbn_2022.behavior_only_nwb_writer.schemas \
    import VBN2022BehaviorOnlyWriterSchema
from allensdk.brain_observatory.behavior.data_objects.metadata.\
    behavior_metadata.date_of_acquisition import DateOfAcquisition
from allensdk.core.authentication import DbCredentials
from allensdk.internal.api import db_connection_creator
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession


class VBN2022BehaviorOnlyWriter(argschema.ArgSchemaParser):
    default_schema = VBN2022BehaviorOnlyWriterSchema

    def run(self):
        """Write behavior only NWB files for input behavior_session_ids from
        using already produced behavior session metadata.
        """
        behavior_session_ids = self.args['behavior_session_id_list']
        output_path = Path(self.args['nwb_output_dir'])
        behavior_session_table = pd.read_csv(
            self.args['behavior_session_table']).set_index('behavior_session_id')
        

        self.logger.info(
            f"\n\nCreating NWB files for {len(behavior_session_ids)} "
            "sessions...\n")
        self.logger.info(
            f"\tWriting files to {output_path}...")

        lims2cred = DbCredentials(dbname='lims2',
                                  user=self.args['lims_user'],
                                  host='limsdb2',
                                  port=5432,
                                  password=self.args['lims_password'])
        db_conn = db_connection_creator(lims2cred)

        passed = 0
        failed = 0
        for bs_id in behavior_session_ids:
            file_path = output_path / f'behavior_session_{bs_id}.nwb'
            if not self.args['clobber'] and file_path.exists():
                continue
            # Date of acquisition is unreliable from LIMS, hence we pull it
            # from the already produced metadata.
            try:
                daq = DateOfAcquisition(
                    datetime.strptime(
                        behavior_session_table.loc[bs_id,
                        'date_of_acquisition'],
                        "%Y-%m-%d %H:%M:%S.%f"))
            except ValueError:
                daq = DateOfAcquisition(
                    datetime.strptime(
                        behavior_session_table.loc[bs_id,
                        'date_of_acquisition'],
                        "%Y-%m-%d %H:%M:%S"))

            session = BehaviorSession.from_lims(
                behavior_session_id=bs_id,
                lims_db=db_conn,
                date_of_acquisition=daq)
            # Edit the value for age using the info in the
            # behavior sessions table. This is due to LIMS
            # containing unreliable info for DAQ and AGE.
            session._metadata._subject_metadata._age._value = \
                behavior_session_table.loc[bs_id, 'age_in_days']

            with pynwb.NWBHDF5IO(file_path, 'w') as nwb_writer:
                nwb_writer.write(session.to_nwb())

        self.logger.info(f"Completed processing. Successful on {passed} "
                         f"session. Failed on {failed} session.")
