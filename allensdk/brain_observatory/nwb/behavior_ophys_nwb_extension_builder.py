from allensdk.brain_observatory.behavior.schemas import (
    OphysBehaviorMetadataSchema,
    BehaviorTaskParametersSchema,
    SubjectMetadataSchema, OphysEyeTrackingRigMetadataSchema
)
from allensdk.brain_observatory.nwb.metadata import (
    create_pynwb_extension_from_schemas
)

if __name__ == "__main__":

    # Run this module to regenerate the extension yaml files into this dir:
    prefix = 'ndx-aibs-behavior-ophys'
    schemas = [BehaviorTaskParametersSchema, SubjectMetadataSchema,
               OphysBehaviorMetadataSchema, OphysEyeTrackingRigMetadataSchema]
    create_pynwb_extension_from_schemas(schemas, prefix)
