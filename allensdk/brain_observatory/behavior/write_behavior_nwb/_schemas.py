from enum import Enum

from argschema import ArgSchema
from argschema.fields import (LogLevel, String, Int, Nested, List)
import marshmallow as mm
import pandas as pd

from allensdk.brain_observatory.argschema_utilities import (
    check_read_access, check_write_access_overwrite, RaisingSchema)


class STIMULUS_NAME(Enum):
    OPHYS_7_receptive_field_mapping = 'OPHYS_7_receptive_field_mapping'


class BehaviorSessionData(RaisingSchema):
    behavior_session_id = Int(required=True,
                              description=("Unique identifier for the "
                                           "behavior session to write into "
                                           "NWB format"))
    foraging_id = String(required=True,
                         description=("The foraging_id for the behavior "
                                      "session"))
    driver_line = List(String,
                       required=True,
                       description='Genetic driver line(s) of subject')
    reporter_line = List(String,
                         required=True,
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
    behavior_stimulus_file = String(required=True,
                                    validate=check_read_access,
                                    description=("Path of behavior_stimulus "
                                                 "camstim *.pkl file"))
    date_of_birth = String(required=True, description="Subject date of birth")
    sex = String(required=True, description="Subject sex")
    age = String(required=True, description="Subject age")
    stimulus_name = String(required=True,
                           description=("Name of stimulus presented during "
                                        "behavior session"))

    @mm.pre_load
    def set_stimulus_name(self, data, **kwargs):
        if data.get("stimulus_name") is None:
            stimulus_file = data["behavior_stimulus_file"]
            pkl = pd.read_pickle(stimulus_file)

            items = pkl['items']

            # Due to historical reasons, most sessions have stimulus name
            # stored Under the "behavior" key but "session 7" sessions have it
            # stored under "foraging"
            params = items.get('behavior', items.get('foraging'))['cl_params']

            if not params:
                raise mm.ValidationError(
                    f'The stimulus file is missing cl_params. '
                    f'Cannot create NWB file.'
                )

            data["stimulus_name"] = params['stage']

            self.__validate_stimulus_name(data=data)

        return data

    @staticmethod
    def __validate_stimulus_name(data):
        session_7 = STIMULUS_NAME.OPHYS_7_receptive_field_mapping.value
        if data['stimulus_name'] == session_7:
            behavior_session_id = data['behavior_session_id']
            raise mm.ValidationError(
                f'Behavior session id {behavior_session_id} is {session_7}. '
                f'Cannot create NWB file because it is missing behavior data.'
            )


class InputSchema(ArgSchema):
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
    output_path = String(required=True,
                         validate=check_write_access_overwrite,
                         description='Path of output.json to be written')
