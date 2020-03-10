import argschema as ags

class runner_config(ags.ArgSchema):
    manifest_file = ags.fields.InputFile(description=".json configurations for running the simulations")
    axon_type = ags.fields.Str(description="axon replacement for the biophysical models",default=None,allow_none=True)

