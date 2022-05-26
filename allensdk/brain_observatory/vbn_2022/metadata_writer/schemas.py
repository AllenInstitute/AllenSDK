import argschema
from argschema.schemas import DefaultSchema
import pathlib
from marshmallow import post_load
from marshmallow.validate import OneOf

from allensdk.brain_observatory.vbn_2022.utils.schemas import (
    ProbeToSkip)


class VBN2022MetadataWriterInputSchema(argschema.ArgSchema):

    ecephys_session_id_list = argschema.fields.List(
            argschema.fields.Int,
            required=True,
            description=("List of ecephys_sessions.id values "
                         "of sessions to be released"))

    probes_to_skip = argschema.fields.List(
            argschema.fields.Nested(ProbeToSkip),
            required=False,
            default=None,
            allow_none=True,
            description=("List of probes to skip"))

    output_dir = argschema.fields.OutputDir(
            required=True,
            description=("Directory where outputs will be written"))

    clobber = argschema.fields.Boolean(
            default=False,
            description=("If false, throw an error if output files "
                         "already exist"))

    ecephys_nwb_dir = argschema.fields.InputDir(
            required=True,
            allow_none=False,
            description=("The directory where ecephys_nwb sessions are "
                         "to be found"))

    ecephys_nwb_prefix = argschema.fields.Str(
        required=False,
        default='ecephys_session',
        description=(
          "Ecephys session NWB files will be looked for "
          "in the form "
          "{ecephys_nwb_dir}/{ecephys_nwb_prefix}_{ecephys_session_id}.nwb")
    )

    on_missing_file = argschema.fields.Str(
            default='error',
            required=False,
            validation=OneOf(('error', 'warn', 'skip')),
            description=("What to do if an input datafile is missing. "
                         "If 'error', raise an exception. "
                         "If 'warn', assign a dummy ID and issue a warning. "
                         "If 'skip', do not list in metadata and issue a "
                         "warning (note, any sessions thus skipped will still "
                         "show up in aggregate metadata; there just will "
                         "be no line for those sessions in tables that list "
                         "data files for release, like sessions.csv)."))

    @post_load
    def validate_paths(self, data, **kwargs):
        fname_lookup = {'units_path': 'units.csv',
                        'channels_path': 'channels.csv',
                        'probes_path': 'probes.csv',
                        'ecephys_sessions_path': 'ecephys_sessions.csv',
                        'behavior_sessions_path': 'behavior_sessions.csv'}

        out_dir = pathlib.Path(data['output_dir'])
        msg = ""
        for fname_k in fname_lookup.keys():
            full_path = out_dir / fname_lookup[fname_k]
            if full_path.exists() and not data['clobber']:
                msg += f"{full_path.resolve().absolute()}\n"
            data[fname_k] = str(full_path.resolve().absolute())

        if len(msg) > 0:
            raise RuntimeError(
                "The following files already exist\n"
                f"{msg}"
                "Run with clobber=True if you want to overwrite")
        return data


class PipelineMetadataSchema(DefaultSchema):

    name = argschema.fields.Str(
            required=True,
            allow_none=False,
            description=(
                "Name of the pipeline component (e.g. 'AllenSDK')"))

    version = argschema.fields.Str(
            required=True,
            allow_none=False,
            description=(
                "Semantic version of the pipeline component"))

    comment = argschema.fields.Str(
            required=False,
            default="",
            description=(
                "Optional comment about this piece of software"))


class VBN2022MetadataWriterOutputSchema(argschema.ArgSchema):

    metadata_files = argschema.fields.List(
            argschema.fields.InputFile,
            description=(
                "Paths to the metadata .csv files "
                "written by this modules"))

    data_pipeline_metadata = argschema.fields.Nested(
            PipelineMetadataSchema,
            many=True,
            description=(
                "Metadata about the pipeline used "
                "to create this data release"))

    project_name = argschema.fields.Str(
            required=True,
            default=None,
            allow_none=False,
            description=(
                "The project name to be passed along "
                "to the data_release_tool when uploading "
                "this dataset"))
