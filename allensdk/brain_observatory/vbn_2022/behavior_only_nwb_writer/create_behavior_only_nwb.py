import argschema
import json
import pandas as pd

from allensdk.brain_observatory.vbn_2022.behavior_only_nwb.schemas import (
    VBN2022BehaviorOnlyWriterSchema)


class VBN2022BehaviorOnlyWriter(argschema.ArgSchemaParser):
    default_schema = VBN2022BehaviorOnlyWriterSchema

    def run(self):

        behavior_session_ids = self.args['behavior_session_id_list']
        output_path = Path(self.args['nwb_output_dir'])
        behavior_session_table = pd.read_csv(
            self.args['behavior_session_table'])

        session_specs = results['sessions']
        msg = results['log']
        self.logger.info(
            f"\n\nCreating NWB files for {len(behavior_session_ids)} "
            "sessions...\n")
        self.logger.info(
            f"\tWriting files to {output_path}...")

        for bs_id in behavior_session_ids:
