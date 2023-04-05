import marshmallow as mm
from allensdk.brain_observatory.argschema_utilities import (
    InputFile,
    RaisingSchema,
)
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.behavior_project_cloud_api import (  # noqa: E501
    sanitize_data_columns,
)
from allensdk.brain_observatory.behavior.behavior_project_cache.tables.metadata_table_schemas import (  # noqa: E501
    BehaviorSessionMetadataSchema,
)
from argschema import ArgSchema
from argschema.fields import (
    Int,
    List,
    LogLevel,
    Nested,
    OutputDir,
    OutputFile,
    String,
)


class BaseInputSchema(ArgSchema):
    class Meta:
        unknown = mm.RAISE

    log_level = LogLevel(
        default="INFO", description="Logging level of the module"
    )
    metadata_table = InputFile(
        required=True,
        description="CSV file containing rows of BehaviorSession or "
        "BehaviorOphysExperiment metadata. Must at least contain the "
        "metadata of the session to be written.",
    )
    skip_metadata_key = List(
        String,
        required=False,
        cli_as_single_argument=True,
        description="List of metadata table keys to skip. Can be used to "
        "override known data issues. Example: ['mouse_id']",
        default=[],
    )
    skip_stimulus_file_key = List(
        String,
        required=False,
        cli_as_single_argument=True,
        description="List of stimulus file keys to skip. Can be used to "
        "override known data issues. Example: ['mouse_id']",
        default=[],
    )
    output_dir_path = OutputDir(
        required=True,
        description="Path of output.json to be written",
    )

    def _get_behavior_metadata(self, bs_row):
        """ """
        behavior_session_metadata = {}

        behavior_session_metadata["age_in_days"] = bs_row["age_in_days"]
        behavior_session_metadata["cre_line"] = bs_row["cre_line"]
        behavior_session_metadata["date_of_acquisition"] = bs_row[
            "date_of_acquisition"
        ]
        behavior_session_metadata["driver_line"] = sorted(
            bs_row["driver_line"]
        )
        behavior_session_metadata["equipment_name"] = bs_row["equipment_name"]
        behavior_session_metadata["full_genotype"] = bs_row["full_genotype"]
        behavior_session_metadata["mouse_id"] = bs_row["mouse_id"]
        behavior_session_metadata["project_code"] = bs_row["project_code"]
        behavior_session_metadata["reporter_line"] = bs_row["reporter_line"]
        behavior_session_metadata["session_type"] = bs_row["session_type"]
        behavior_session_metadata["sex"] = bs_row["sex"]

        return behavior_session_metadata


class BehaviorInputSchema(BaseInputSchema):
    behavior_session_id = Int(
        required=True, description="Id of BehaviorSession to create."
    )

    behavior_session_metadata = Nested(
        BehaviorSessionMetadataSchema,
        required=True,
        description="Data pertaining to a behavior session",
    )

    @mm.pre_load
    def retreive_metadata(self, data, **kwargs):
        """Load the data from csv using Pandas the same as the
        project_cloud api.
        """
        df = sanitize_data_columns(
            data["metadata_table"], dtype_convert={"mouse_id": str}
        )
        df.set_index("behavior_session_id", inplace=True)
        try:
            bs_row = df.loc[int(data["behavior_session_id"])]
        except KeyError:
            raise KeyError(
                f"Behavior session id {data['behavior_session_id']} "
                "not in input BehaviorSessionTable. Exiting."
            )

        data["behavior_session_metadata"] = self._get_behavior_metadata(bs_row)
        data["behavior_session_metadata"]["behavior_session_id"] = bs_row.name

        return data


class OutputSchema(RaisingSchema):
    input_parameters = Nested(BehaviorInputSchema)
    output_path = OutputFile(
        required=True,
        description="Path of output NWB file.",
    )
