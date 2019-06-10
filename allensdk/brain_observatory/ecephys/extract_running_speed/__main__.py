
import numpy as np

from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import EcephysSyncDataset

from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
from allensdk.brain_observatory.ecephys.file_io.stim_file import CamStimOnePickleStimFile
from ._schemas import InputParameters, OutputParameters
from .utilities import degrees_to_radians, angular_to_linear_velocity

def extract_running_speeds(args):
    
    stim_file = CamStimOnePickleStimFile.factory(args['stimulus_pkl_path'])
    sync_dataset = EcephysSyncDataset.factory(args['sync_h5_path'])

    radian_angular_wheel_velocity = degrees_to_radians(stim_file.angular_wheel_velocity)
    linear_running_speed = angular_to_linear_velocity(radian_angular_wheel_velocity, args['wheel_radius'])

    frame_times = sync_dataset.extract_frame_times(strategy=args['frame_time_strategy'])
    
    np.save(args['output_timestamps_path'], frame_times, allow_pickle=False)
    np.save(args['output_running_speeds_path'], linear_running_speed, allow_pickle=False)

    return {
        'output_running_speeds_path': args['output_running_speeds_path'],
        'output_timestamps_path': args['output_timestamps_path']
    }


def main():

    mod = ArgSchemaParserPlus(schema_type=InputParameters, output_schema_type=OutputParameters)
    output = extract_running_speeds(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))

    
if __name__ == "__main__":
    main()

