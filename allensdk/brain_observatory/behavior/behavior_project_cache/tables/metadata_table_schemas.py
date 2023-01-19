import marshmallow as mm
import pandas as pd
from allensdk.brain_observatory.argschema_utilities import RaisingSchema
from argschema.fields import Int, List, String

"""Schemas for validating data in the metadata table. Used in NWB creation
to validating typing against expected values in the session objects.
"""


class BehaviorSessionMetadataSchema(RaisingSchema):
    age_in_days = Int(required=True, description="Subject age")
    behavior_session_id = Int(
        required=True,
        description=(
            "Unique identifier for the "
            "behavior session to write into "
            "NWB format"
        ),
    )
    cre_line = String(
        required=True, description="Genetic cre line of the subject."
    )
    date_of_acquisition = String(
        required=True,
        description=(
            "Date of acquisition of " "behavior session, in string " "format"
        ),
    )
    driver_line = List(
        String,
        required=True,
        cli_as_single_argument=True,
        description="Genetic driver line(s) of subject",
    )
    equipment_name = String(
        required=True, description=("Name of the equipment used.")
    )
    full_genotype = String(
        required=True, description="Full genotype of subject"
    )
    mouse_id = String(
        required=True,
        description="LabTracks ID of the subject. aka external_specimen_name.",
    )
    project_code = String(
        rquired=True,
        description="LabTracks ID of the subject. aka external_specimen_name.",
    )
    reporter_line = String(
        required=True, description="Genetic reporter line(s) of subject"
    )
    session_type = String(
        required=True, description="Full name of session type."
    )
    sex = String(required=True, description="Subject sex")

    @mm.post_load
    def convert_date_time(self, data, **kwargs):
        """Change date_of_acquisition to a date time type from string."""
        data["date_of_acquisition"] = pd.to_datetime(
            data["date_of_acquisition"], utc=True
        )
        return data


class OphysExperimentMetadataSchema(BehaviorSessionMetadataSchema):
    imaging_depth = Int(
        required=True, description="Imaging depth of the OphysExperiment."
    )
    imaging_plane_group = Int(
        required=True,
        allow_none=True,
        description="Imaging plane group of OphysExperiment.",
    )
    indicator = String(required=True, description="String indicator line.")
    ophys_container_id = Int(
        required=True,
        description="ID of ophys container of which this experiment is a "
        "member.",
    )
    ophys_experiment_id = Int(
        required=True, description="ID of the ophys experiment."
    )
    ophys_session_id = Int(
        required=True,
        description="ID of the ophys session this experiment is a member of.",
    )
    targeted_imaging_depth = Int(
        required=True,
        description="Average of all experiments in the container.",
    )
    targeted_structure = String(
        required=True, description="String name of the structure targeted."
    )
