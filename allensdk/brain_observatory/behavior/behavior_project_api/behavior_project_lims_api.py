from pathlib import Path
import shutil
import warnings

import pandas as pd

from .behavior_project_api import BehaviorProjectApi
from allensdk.brain_observatory.ecephys.ecephys_project_api.http_engine import HttpEngine
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import postgres_macros, build_and_execute
from allensdk.brain_observatory.behavior.metadata_processing import get_task_parameters
from allensdk.internal.api import PostgresQueryMixin

class BehaviorProjectLimsApi(BehaviorProjectApi):
    def __init__(self, postgres_engine, app_engine):
        self.postgres_engine = postgres_engine
        self.app_engine = app_engine

    def get_session_nwb(self, experiment_id):
        nwb_response = build_and_execute(
            """
            SELECT 
            wkf.id, wkf.storage_directory, wkf.attachable_id, wkf.filename
            FROM ophys_experiments oe
            LEFT JOIN well_known_files wkf 
                ON wkf.attachable_id=oe.id
            LEFT JOIN well_known_file_types wkt
                ON wkf.well_known_file_type_id = wkt.id
            WHERE wkt.name = 'BehaviorOphysNwb'
            AND oe.id = {{experiment_id}}
            """,
            engine=self.postgres_engine.select,
            experiment_id=experiment_id,
        )

        if nwb_response.shape[0] != 1:
            raise ValueError(
                f"expected exactly 1 current NWB file for session {experiment_id}, "
                f"found {nwb_response.shape[0]}: {pd.DataFrame(nwb_response)}"
            )

        nwb_id = nwb_response.loc[0, "id"]
        return self.app_engine.stream(
            f"well_known_files/download/{nwb_id}?wkf_id={nwb_id}"
        )

    def get_sessions(
        self,
        container_ids=None,
        container_workflow_states=("container_qc","postprocessing","complete"),
        project_names=("Visual Behavior production",),
        reporter_lines=('Ai148(TIT2L-GC6f-ICL-tTA2)', 'Ai93(TITL-GCaMP6f)',),
        filter_failed_experiments=True,
        **kwargs
    ):

        response = build_and_execute(
            """
                {%- import 'postgres_macros' as pm -%}
                {%- import 'macros' as m -%}

                SELECT
                oec.visual_behavior_experiment_container_id as container_id,
                oec.ophys_experiment_id,
                vbc.workflow_state as container_workflow_state,
                oe.workflow_state as experiment_workflow_state,
                os.date_of_acquisition,
                d.full_genotype as full_genotype,
                rl.reporter_line,
                dl.driver_line,
                d.id as donor_id,
                genders.name as sex,
                id.depth as imaging_depth,
                st.acronym as targeted_structure,
                os.name as session_name,
                os.foraging_id,
                equipment.name as equipment_name,
                pr.name as project_name

                FROM ophys_experiments_visual_behavior_experiment_containers oec
                JOIN visual_behavior_experiment_containers vbc 
                    ON oec.visual_behavior_experiment_container_id = vbc.id
                JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id = os.id
                JOIN projects pr ON pr.id = os.project_id
                JOIN specimens sp ON sp.id=os.specimen_id
                JOIN donors d ON d.id=sp.donor_id

                JOIN (
                    SELECT g.name as reporter_line, d.id as donor_id
                    FROM donors d
                    LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
                    LEFT JOIN genotypes g ON g.id=dg.genotype_id
                    LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
                    WHERE gt.name='reporter'
                ) rl ON rl.donor_id = d.id

                JOIN (
                    SELECT ARRAY_AGG (g.name) as driver_line, d.id as donor_id
                    FROM donors d
                    LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
                    LEFT JOIN genotypes g ON g.id=dg.genotype_id
                    LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
                    WHERE gt.name='driver'
                    GROUP BY d.id
                ) dl ON dl.donor_id = d.id

                JOIN genders ON genders.id = d.gender_id
                JOIN imaging_depths id ON id.id=os.imaging_depth_id
                JOIN structures st ON st.id=oe.targeted_structure_id
                JOIN equipment ON equipment.id=os.equipment_id

                WHERE TRUE
                {{pm.optional_contains('oec.visual_behavior_experiment_container_id', container_ids) -}}
                {{pm.optional_contains('vbc.workflow_state', container_workflow_states, True) -}}
                {{pm.optional_contains('pr.name', project_names, True) -}}
                {{pm.optional_contains('rl.reporter_line', reporter_lines, True) -}}
            """,
            base=postgres_macros(),
            engine=self.postgres_engine.select,
            container_ids=container_ids,
            container_workflow_states=container_workflow_states,
            reporter_lines=reporter_lines,
            project_names=project_names,
        )

        # Need to get the mtrain stage for each recording session
        foraging_ids = response['foraging_id'][~pd.isnull(response['foraging_id'])]
        mtrain_api = PostgresQueryMixin(dbname="mtrain", user="mtrainreader", host="prodmtrain1", password="mtrainro", port=5432)
        query = """
                SELECT
				stages.name as stage_name, 
				bs.id as foraging_id
				FROM behavior_sessions bs
				LEFT JOIN states ON states.id = bs.state_id
				LEFT JOIN stages ON stages.id = states.stage_id
                WHERE bs.id IN ({})
            """.format(",".join(["'{}'".format(x) for x in foraging_ids]))
        mtrain_response = pd.read_sql(query, mtrain_api.get_connection())
        response = response.merge(mtrain_response, on='foraging_id', how='left')

        # Need to determine the retake number for each type of session
        mouse_gb = response.groupby('donor_id')
        unique_mice = response['donor_id'].unique()
        for mouse_id in unique_mice:
            mouse_df = mouse_gb.get_group(mouse_id)
            stage_gb = mouse_df.groupby('stage_name')
            unique_stages = mouse_df['stage_name'][~pd.isnull(mouse_df['stage_name'])].unique()
            for stage_name in unique_stages:
                # Iterate through the sessions sorted by date and save the index to the row
                sessions_this_stage = stage_gb.get_group(stage_name).sort_values('date_of_acquisition')
                for ind_enum, (ind_row, row) in enumerate(sessions_this_stage.iterrows()):
                    response.at[ind_row, 'retake_number'] = ind_enum

        # Failed sessions are needed to calculate the retake number, so we can't filter them out
        # in the LIMS query.
        if filter_failed_experiments:
            response = response.query("experiment_workflow_state == 'passed'")

        return response

    @classmethod
    def default(cls, pg_kwargs=None, app_kwargs=None):

        _pg_kwargs = {}
        if pg_kwargs is not None:
            _pg_kwargs.update(pg_kwargs)

        _app_kwargs = {"scheme": "http", "host": "lims2"}
        if app_kwargs is not None:
            _app_kwargs.update(app_kwargs)

        pg_engine = PostgresQueryMixin(**_pg_kwargs)
        app_engine = HttpEngine(**_app_kwargs)
        return cls(pg_engine, app_engine)

def parse_passive(behavior_stage):
    '''
    Args:
        behavior_stage (str): the stage string, e.g. OPHYS_1_images_A or OPHYS_1_images_A_passive
    Returns:
        passive (bool): whether or not the session was a passive session
    '''
    r = re.compile(".*_passive")
    if r.match(behavior_stage):
        return True
    else:
        return False


def parse_image_set(behavior_stage):
    '''
    Args:
        behavior_stage (str): the stage string, e.g. OPHYS_1_images_A or OPHYS_1_images_A_passive
    Returns:
        image_set (str): which image set is designated by the stage name
    '''
    r = re.compile(".*images_(?P<image_set>[A-Z]).*")
    image_set = r.match(behavior_stage).groups('image_set')[0]
    return image_set
