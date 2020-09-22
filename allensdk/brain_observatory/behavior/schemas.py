from marshmallow import Schema, fields, RAISE
import numpy as np


STYPE_DICT = {fields.Float: 'float', fields.Int: 'int',
              fields.String: 'text', fields.List: 'text',
              fields.DateTime: 'text', fields.UUID: 'text'}
TYPE_DICT = {fields.Float: float, fields.Int: int, fields.String: str,
             fields.List: np.ndarray, fields.DateTime: str, fields.UUID: str}


class RaisingSchema(Schema):
    class Meta:
        unknown = RAISE


class SubjectMetadataSchema(RaisingSchema):
    """This schema contains metadata pertaining to a subject in either a
    behavior or behavior + ophys experiment.
    """

    neurodata_type = 'BehaviorSubject'
    neurodata_type_inc = 'Subject'
    neurodata_doc = "Metadata for an AIBS behavior or behavior + ophys subject"

    age = fields.String(
        doc='Age of the specimen donor/subject',
        required=True,
    )
    driver_line = fields.List(
        fields.String,
        doc="Driver line of subject",
        required=True,
        shape=(None,),
    )
    # 'full_genotype' will be stored in pynwb Subject 'genotype' attr
    genotype = fields.String(
        doc='full genotype of subject',
        required=True,
    )
    # 'LabTracks_ID' will be stored in pynwb Subject 'subject_id' attr
    subject_id = fields.Int(
        doc='LabTracks ID of subject',
        required=True,
    )
    reporter_line = fields.List(
        fields.String,
        doc="Reporter line of subject",
        required=True,
        shape=(None,),
    )
    sex = fields.String(
        doc='Sex of the specimen donor/subject',
        required=True,
    )


class BehaviorMetadataSchema(RaisingSchema):
    """This schema contains metadata pertaining to behavior.
    """

    behavior_session_uuid = fields.UUID(
        doc='MTrain record for session, also called foraging_id',
        required=True,
    )
    stimulus_frame_rate = fields.Float(
        doc=('Frame rate (frames/second) of the '
             'visual_stimulus from the monitor'),
        required=True,
    )


class OphysMetadataSchema(RaisingSchema):
    """This schema contains metadata pertaining to optical physiology (ophys).
    """
    emission_lambda = fields.Float(
        doc='emission_lambda',
        required=True,
    )
    excitation_lambda = fields.Float(
        doc='excitation_lambda',
        required=True,
    )
    experiment_container_id = fields.Int(
        doc='Container ID for the container that contains this ophys session',
        required=True,
    )
    imaging_depth = fields.Int(
        doc=('Depth (microns) below the cortical surface '
             'targeted for two-photon acquisition'),
        required=True,
    )
    indicator = fields.String(
        doc='indicator',
        required=True,
    )
    ophys_experiment_id = fields.Int(
        doc='Id for this ophys session',
        required=True,
    )
    ophys_frame_rate = fields.Float(
        doc='Frame rate (frames/second) of the two-photon microscope',
        required=True,
    )
    rig_name = fields.String(
        doc='name of two-photon rig',
        required=True,
    )
    targeted_structure = fields.String(
        doc='Anatomical structure targeted for two-photon acquisition',
        required=True,
    )


class OphysBehaviorMetadataSchema(BehaviorMetadataSchema, OphysMetadataSchema):
    """ This schema contains fields pertaining to ophys+behavior. It is used
    as a template for generating our custom NWB behavior + ophys extension.
    """

    neurodata_type = 'OphysBehaviorMetadata'
    neurodata_type_inc = 'LabMetaData'
    neurodata_doc = "Metadata for behavior + ophys experiments"

    session_type = fields.String(
        doc='Experimental session description',
        allow_none=True,
        required=True,
    )
    experiment_datetime = fields.DateTime(
        doc='Date of the experiment (UTC, as string)',
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


class CompleteOphysBehaviorMetadataSchema(OphysBehaviorMetadataSchema,
                                          SubjectMetadataSchema):
    """This schema combines fields from behavior, ophys, and subject schemas.
    Metadata info is passed by the behavior+ophys session in a combined lump
    containing all the field types.
    """
    pass


class BehaviorTaskParametersSchema(RaisingSchema):
    """This schema encompasses task parameters used for behavior or
    ophys + behavior.
    """
    neurodata_type = 'BehaviorTaskParameters'
    neurodata_type_inc = 'LabMetaData'
    neurodata_doc = "Metadata for behavior or behavior + ophys task parameters"

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
        doc='stimulus',
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
