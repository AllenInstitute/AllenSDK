from marshmallow import Schema, fields, RAISE
import numpy as np


STYPE_DICT = {fields.Float: 'float', fields.Int: 'int', fields.String: 'text', fields.List: 'text', fields.DateTime: 'text', fields.UUID: 'text'}
TYPE_DICT = {fields.Float: float, fields.Int: int, fields.String: str, fields.List: np.ndarray, fields.DateTime: str, fields.UUID: str}


class RaisingSchema(Schema):
    class Meta:
        unknown = RAISE


class OphysBehaviorMetaDataSchema(RaisingSchema):
    """ base schema for all timeseries
    """

    neurodata_type = 'OphysBehaviorMetaData'

    ophys_experiment_id = fields.Int(
        doc='Id for this ophys session',
        required=True,
    )

    experiment_container_id = fields.Int(
        doc='Container ID for the container that contains this ophys session',
        required=True,
    )

    ophys_frame_rate = fields.Float(
        doc='Frame rate (frames/second) of the two-photon microscope',
        required=True,
    )

    stimulus_frame_rate = fields.Float(
        doc='Frame rate (frames/second) of the visual_stimulus from the monitor',
        required=True,
    )

    targeted_structure = fields.String(
        doc='Anatomical structure targeted for two-photon acquisition',
        required=True,
    )

    imaging_depth = fields.Int(
        doc='Depth (microns) below the cortical surface targeted for two-photon acquisition',
        required=True,
    )

    session_type = fields.String(
        doc='Experimental session description',
        allow_none=True,
        required=True,
    )

    experiment_datetime = fields.DateTime(
        doc='Date of the experiment (UTC, as string)',
        required=True,
    )

    reporter_line = fields.List(
        fields.String,
        doc='Reporter line',
        required=True,
        shape=(None,),
    )    

    driver_line = fields.List(
        fields.String,
        doc='Driver line',
        required=True,
        shape=(None,),
    )

    LabTracks_ID = fields.Int(
        doc='LabTracks ID of subject',
        required=True,
    )

    full_genotype = fields.String(
        doc='full genotype of subject',
        required=True,
    )

    behavior_session_uuid = fields.UUID(
        doc='MTrain record for session, also called foraging_id',
        required=True,
    )

    rig_name = fields.String(
        doc='name of two-photon rig',
        required=True,
    )

    excitation_lambda = fields.Float(
        doc='excitation_lambda',
        required=True,
    )

    emission_lambda = fields.Float(
        doc='emission_lambda',
        required=True,
    )

    indicator = fields.String(
        doc='indicator',
        required=True,
    )

    field_of_view_width = fields.Int(
        doc='field_of_view_width',
        required=True,
    )

    field_of_view_height = fields.Int(
        doc='field_of_view_height',
        required=True,
    )

    sex = fields.String(
        doc='Sex of the specimen doner/subject',
        required=True,
    )

    age = fields.String(
        doc='Age of the specimen doner/subject',
        required=True,
    )


class OphysBehaviorTaskParametersSchema(RaisingSchema):
    """ base schema for all timeseries
    """

    neurodata_type = 'OphysBehaviorTaskParameters'

    blank_duration_sec = fields.List(
        fields.Float,
        doc='blank duration in seconds',
        required=True,
        shape=(2,),
    )

    stimulus_duration_sec = fields.Float(
        doc='duration of each stimulus presentation in seconds',
        required=True,
    )

    omitted_flash_fraction = fields.Float(
        doc='omitted_flash_fraction',
        required=True,
        allow_nan=True,
    )

    response_window_sec = fields.List(
        fields.Float,
        doc='response_window in seconds',
        required=True,
        shape=(2,),
    )

    reward_volume = fields.Float(
        doc='reward_volume',
        required=True,
    )

    stage = fields.String(
        doc='stage',
        required=True,
    )

    stimulus = fields.String(
        doc='stage',
        required=True,
    )

    stimulus_distribution = fields.String(
        doc='stimulus_distribution',
        required=True,
    )

    task = fields.String(
        doc='task',
        required=True,
    )

    n_stimulus_frames = fields.Int(
        doc='n_stimulus_frames',
        required=True,
    )
