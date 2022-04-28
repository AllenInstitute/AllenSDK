import argschema
import pathlib
import re
from marshmallow import post_load


class ProbeToSkip(argschema.ArgSchema):

    session = argschema.fields.Int(
            required=True,
            description=("The ecephys_session_id associated with "
                         "the bad probe"))

    probe = argschema.fields.Str(
            required=True,
            description=("The name of the bad probe, e.g. 'probeA'"))

    @post_load
    def validate_probe_names(self, data, **kwargs):
        pattern = re.compile('probe[A-Z]')
        match = pattern.match(data['probe'])
        if match is None or len(data['probe']) != 6:
            raise ValueError(
                f"{data['probe']} is not a valid probe name; "
                "must be like 'probe[A-Z]'")
        return data


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

    @post_load
    def validate_paths(self, data, **kwargs):
        fname_lookup = {'units_path': 'units.csv',
                        'channels_path': 'channels.csv',
                        'probes_path': 'probes.csv',
                        'ecephys_sessions_path': 'sessions.csv',
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
