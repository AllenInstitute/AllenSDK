from argschema import ArgSchema, InputFile
from argschema.fields import LogLevel, String, Int, Nested, \
    Boolean, \
    Float, List, Dict
from marshmallow import RAISE

from allensdk.brain_observatory.argschema_utilities import check_read_access, \
    check_write_access_overwrite, RaisingSchema


class CellSpecimenTable(RaisingSchema):
    cell_roi_id = Dict(String, Int, required=True)
    cell_specimen_id = Dict(String, Int(allow_none=True), required=True)
    x = Dict(String, Int, required=True)
    y = Dict(String, Int, required=True)
    max_correction_up = Dict(String, Float, required=True)
    max_correction_right = Dict(String, Float, required=True)
    max_correction_down = Dict(String, Float, required=True)
    max_correction_left = Dict(String, Float, required=True)
    valid_roi = Dict(String, Boolean, required=True)
    height = Dict(String, Int, required=True)
    width = Dict(String, Int, required=True)
    mask_image_plane = Dict(String, Int, required=True)
    roi_mask = Dict(String, List(List(Boolean)), required=True)


class SessionData(RaisingSchema):
    ophys_experiment_id = Int(required=True,
                              description='unique identifier for this ophys '
                                          'session')
    ophys_session_id = Int(required=True,
                           description='The ophys session id that the ophys '
                                       'experiment to be written to NWB is '
                                       'from')
    behavior_session_id = Int(required=True,
                              description='The behavior session id that the '
                                          'ophys experiment to be written to '
                                          'written to NWB is from')
    foraging_id = String(required=True,
                         description='The foraging id associated with the '
                                     'ophys session')
    rig_name = String(required=True, description='name of ophys device')
    movie_height = Int(required=True,
                       description='height of field-of-view for 2p movie')
    movie_width = Int(required=True,
                      description='width of field-of-view for 2p movie')
    container_id = Int(required=True,
                       description='container that this experiment is in')
    sync_file = String(required=True, description='path to sync file')
    max_projection_file = String(required=True,
                                 description='path to max_projection file')
    behavior_stimulus_file = String(required=True,
                                    description='path to behavior_stimulus '
                                                'file')
    dff_file = String(required=True, description='path to dff file')
    demix_file = String(required=True, description='path to demix file')
    average_intensity_projection_image_file = String(
        required=True,
        description='path to '
                    'average_intensity_projection_image file')
    rigid_motion_transform_file = String(required=True,
                                         description='path to '
                                                     'rigid_motion_transform'
                                                     ' file')
    targeted_structure = String(required=True,
                                description='Anatomical structure that the '
                                            'experiment targeted')
    targeted_depth = Int(required=True,
                         description='Cortical depth that the experiment '
                                     'targeted')
    stimulus_name = String(required=True, description='Stimulus Name')
    date_of_acquisition = String(required=True,
                                 description='date of acquisition of '
                                             'experiment, as string (no '
                                             'timezone info but relative ot '
                                             'UTC)')
    reporter_line = List(String, required=True, description='reporter line')
    driver_line = List(String, required=True, description='driver line')
    external_specimen_name = Int(required=True,
                                 description='LabTracks ID of the animal')
    full_genotype = String(required=True, description='full genotype')
    surface_2p_pixel_size_um = Float(required=True,
                                     description='the spatial extent (in um) '
                                                 'of the 2p field-of-view')
    ophys_cell_segmentation_run_id = Int(required=True,
                                         description='ID of the active '
                                                     'segmentation run used '
                                                     'to generate this file')
    cell_specimen_table_dict = Nested(CellSpecimenTable, required=True,
                                      description='Table of cell specimen '
                                                  'info')
    sex = String(required=True, description='sex')
    age = String(required=True, description='age')
    eye_tracking_rig_geometry = Dict(
        required=True,
        description="Mapping containing information about session rig "
                    "geometry used for eye gaze mapping."
    )
    eye_tracking_filepath = String(
        required=True,
        validate=check_read_access,
        description="h5 filepath containing eye tracking ellipses"
    )
    events_file = InputFile(
        required=True,
        description='h5 filepath to events data'
    )
    imaging_plane_group = Int(
        required=True,
        allow_none=True,
        description="A numeric index that indicates the order that the "
                    "frames were acquired when dealing with an imaging plane "
                    "in a mesoscope experiment. Will be None for Scientifica "
                    "experiments."
    )
    plane_group_count = Int(
        required=True,
        description="The total number of plane groups associated with the "
                    "ophys session that the experiment belongs to. Will be 0 "
                    "for Scientifica experiments and nonzero for Mesoscope "
                    "experiments."
    )


class InputSchema(ArgSchema):
    class Meta:
        unknown = RAISE

    log_level = LogLevel(default='INFO',
                         description='set the logging level of the module')
    session_data = Nested(SessionData, required=True,
                          description='records of the individual probes '
                                      'used for this experiment')
    output_path = String(required=True, validate=check_write_access_overwrite,
                         description='write outputs to here')
    skip_eye_tracking = Boolean(
        required=True, default=False,
        description="Whether or not to skip processing eye tracking data. "
                    "If True, no eye tracking data will be written to NWB")


class OutputSchema(RaisingSchema):
    input_parameters = Nested(InputSchema)
    output_path = String(required=True, description='write outputs to here')
