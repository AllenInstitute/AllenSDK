import pathlib

import argschema
from allensdk.brain_observatory.vbn_2022.utils.schemas import ProbeToSkip
from argschema.schemas import DefaultSchema
from marshmallow import post_load
from marshmallow.validate import OneOf


class VBN2022MetadataWriterInputSchema(argschema.ArgSchema):

    ecephys_session_id_list = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        description=(
            "List of ecephys_sessions.id values " "of sessions to be released"
        ),
    )

    failed_ecephys_session_id_list = argschema.fields.List(
        argschema.fields.Int,
        required=False,
        default=None,
        allow_none=True,
        description=(
            "List of ecephys_sessions.id values "
            "associated with this release that were "
            "failed. These are required to "
            "self-consistently construct the history of "
            "each mouse passing through the apparatus."
        ),
    )

    probes_to_skip = argschema.fields.List(
        argschema.fields.Nested(ProbeToSkip),
        required=False,
        default=None,
        allow_none=True,
        description=("List of probes to skip"),
    )

    output_dir = argschema.fields.OutputDir(
        required=True, description=("Directory where outputs will be written")
    )

    clobber = argschema.fields.Boolean(
        default=False,
        description=(
            "If false, throw an error if output files " "already exist"
        ),
    )

    ecephys_nwb_dir = argschema.fields.InputDir(
        required=True,
        allow_none=False,
        description=(
            "The directory where ecephys_nwb sessions are " "to be found"
        ),
    )
    behavior_nwb_dir = argschema.fields.InputDir(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "The directory where behavior_nwb sessions are "
            "to be found. Default to the value of ecephys_nwb "
            "if not set/set to None."
        ),
    )

    ecephys_nwb_prefix = argschema.fields.Str(
        required=False,
        default="ecephys_session",
        description=(
            "Ecephys session NWB files will be looked for "
            "in the form "
            "{ecephys_nwb_dir}/{ecephys_nwb_prefix}_{ecephys_session_id}.nwb"
        ),
    )
    behavior_nwb_prefix = argschema.fields.Str(
        required=False,
        default="behavior_session",
        description=(
            "Behavior session NWB files will be looked for "
            "in the form "
            "{behavior_nwb_dir}/"
            "{behavior_nwb_prefix}_{behavior_session_id}.nwb"
        ),
    )

    supplemental_data = argschema.fields.List(
        argschema.fields.Dict,
        default=None,
        allow_none=True,
        description=(
            "List of dicts definining any supplemental columns "
            "that need to be added to the ecephys_sessions.csv "
            "table. Each dict should represent a row in a dataframe "
            "that will get merged on ecephys_session_id with "
            "the ecephys_sessions table (row must therefore contain "
            "ecephys_session_id)"
        ),
    )

    on_missing_file = argschema.fields.Str(
        default="error",
        required=False,
        validation=OneOf(("error", "warn", "skip")),
        description=(
            "What to do if an input datafile is missing. "
            "If 'error', raise an exception. "
            "If 'warn', assign a dummy ID and issue a warning. "
            "If 'skip', do not list in metadata and issue a "
            "warning (note, any sessions thus skipped will still "
            "show up in aggregate metadata; there just will "
            "be no line for those sessions in tables that list "
            "data files for release, like sessions.csv)."
        ),
    )

    n_workers = argschema.fields.Int(
        default=8,
        allow_none=True,
        description='Number of workers for reading from pkl file. '
                    'Default=8 due to issues with making too many '
                    'requests to the database. Increase if too slow, decrease '
                    'if the database rejects the connection'
    )

    @post_load
    def validate_paths(self, data, **kwargs):
        fname_lookup = {
            "units_path": "units.csv",
            "channels_path": "channels.csv",
            "probes_path": "probes.csv",
            "ecephys_sessions_path": "ecephys_sessions.csv",
            "behavior_sessions_path": "behavior_sessions.csv",
        }

        out_dir = pathlib.Path(data["output_dir"])
        msg = ""
        for fname_k in fname_lookup.keys():
            full_path = out_dir / fname_lookup[fname_k]
            if full_path.exists() and not data["clobber"]:
                msg += f"{full_path.resolve().absolute()}\n"
            data[fname_k] = str(full_path.resolve().absolute())

        if len(msg) > 0:
            raise RuntimeError(
                "The following files already exist\n"
                f"{msg}"
                "Run with clobber=True if you want to overwrite"
            )
        return data


class PipelineMetadataSchema(DefaultSchema):

    name = argschema.fields.Str(
        required=True,
        allow_none=False,
        description=("Name of the pipeline component (e.g. 'AllenSDK')"),
    )

    version = argschema.fields.Str(
        required=True,
        allow_none=False,
        description=("Semantic version of the pipeline component"),
    )

    comment = argschema.fields.Str(
        required=False,
        default="",
        description=("Optional comment about this piece of software"),
    )


class DataReleaseToolsInputSchema(argschema.ArgSchema):
    """
    This schema will be used as the output schema for
    data_release.metadata_writer modules. It is actually
    a subset of the input schema for the
    informatics_data_release_tools (the output of the metadata
    writers is meant to be the input of the data_release_tool)
    """

    metadata_files = argschema.fields.List(
        argschema.fields.InputFile,
        description=(
            "Paths to the metadata .csv files " "written by this modules"
        ),
    )

    data_pipeline_metadata = argschema.fields.Nested(
        PipelineMetadataSchema,
        many=True,
        description=(
            "Metadata about the pipeline used " "to create this data release"
        ),
    )

    project_name = argschema.fields.Str(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "The project name to be passed along "
            "to the data_release_tool when uploading "
            "this dataset"
        ),
    )
