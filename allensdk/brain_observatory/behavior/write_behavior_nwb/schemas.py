import argschema.fields
import marshmallow as mm
import pandas as pd
from argschema import ArgSchema
from argschema.fields import Int, List, LogLevel, Nested, String

from allensdk.brain_observatory.argschema_utilities import (
    RaisingSchema, check_write_access_overwrite)


class BaseBehaviorSessionDataSchema(RaisingSchema):
    behavior_session_id = Int(required=True,
                              description=("Unique identifier for the "
                                           "behavior session to write into "
                                           "NWB format"))
    driver_line = List(String,
                       required=True,
                       cli_as_single_argument=True,
                       description='Genetic driver line(s) of subject')
    reporter_line = List(String,
                         required=True,
                         cli_as_single_argument=True,
                         description='Genetic reporter line(s) of subject')
    full_genotype = String(required=True,
                           description='Full genotype of subject')
    rig_name = String(required=True,
                      description=("Name of experimental rig used for "
                                   "the behavior session"))
    date_of_acquisition = String(required=True,
                                 description=("Date of acquisition of "
                                              "behavior session, in string "
                                              "format"))
    external_specimen_name = Int(required=True,
                                 description='LabTracks ID of the subject')
    behavior_stimulus_file = argschema.fields.InputFile(
        required=True,
        description=("Path of behavior_stimulus "
                     "camstim *.pkl file"))
    date_of_birth = String(required=True, description="Subject date of birth")
    sex = String(required=True, description="Subject sex")
    age = String(required=True, description="Subject age")


class BehaviorSessionData(BaseBehaviorSessionDataSchema):
    stimulus_name = String(required=True,
                           description=("Name of stimulus presented during "
                                        "behavior session"))

    foraging_id = String(required=True,
                         description=("The foraging_id for the behavior "
                                      "session"))

    @mm.pre_load
    def set_stimulus_name(self, data, **kwargs):
        if data.get("stimulus_name") is None:
            pkl = pd.read_pickle(data["behavior_stimulus_file"])
            try:
                stimulus_name = pkl["items"]["behavior"]["cl_params"]["stage"]
            except KeyError:
                raise mm.ValidationError(
                    f"Could not obtain stimulus_name/stage information from "
                    f"the *.pkl file ({data['behavior_stimulus_file']}) "
                    f"for the behavior session to save as NWB! The "
                    f"following series of nested keys did not work: "
                    f"['items']['behavior']['cl_params']['stage']"
                )
            data["stimulus_name"] = stimulus_name
        return data


class BehaviorInputSchema(ArgSchema):
    class Meta:
        unknown = mm.RAISE
    log_level = LogLevel(default='INFO',
                         description='Logging level of the module')
    session_data = Nested(BehaviorSessionData,
                          required=True,
                          description='Data pertaining to a behavior session')
    output_path = String(required=True,
                         validate=check_write_access_overwrite,
                         description='Path of output.json to be written')


class OutputSchema(RaisingSchema):
    input_parameters = Nested(BehaviorInputSchema)
    output_path = String(required=True,
                         validate=check_write_access_overwrite,
                         description='Path of output.json to be written')
