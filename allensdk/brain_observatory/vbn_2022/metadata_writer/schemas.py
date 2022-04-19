import argschema


class VBN2022MetadataWriterInputSchema(argschema.ArgSchema):

    ecephys_session_id_list = argschema.fields.List(
            argschema.fields.Int,
            required=True,
            description=("List of ecephys_sessions.id values "
                         "of sessions to be released"))
