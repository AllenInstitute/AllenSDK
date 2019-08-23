import os
import subprocess
import glob
import pandas as pd
import glob

from create_input_json import createInputJson

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

#available_directories = glob.glob('/mnt/hdd0/RE-SORT/mouse*') 

mice = [#386129,387858,388521,394208,
        #       404553,404568,404551,404571,404555,404569,403407,412791,
        #       412792,405755,
        #       404570,404554,405751,406807,412794,412793,406805,412799,407972,406808,
        #       412802,410343,408152,412803,412804,415149,412809,415148,419117,410315,419114,
        #       419115,
        #       419116,419112,419118,419119,416861,416356,416357,417678,424445,418196,421529,421338,
        #       424448,
        #       425599,425589,432104,425597,432105,
        #       433891,429857,434845,
        #       434494,
        #       429860,434843,
        #       434488,437660,437661,434836,
        #       434838,448503]
        434845, 425589]

#mice = [int(name[-6:]) for name in available_directories] 
# = [int(name) for name in mice] 
  

json_directory = '/mnt/md0/data/json_files'

modules = ['allensdk.brain_observatory.ecephys.align_timestamps', 
           'allensdk.brain_observatory.ecephys.stimulus_table', 
           'allensdk.brain_observatory.extract_running_speed', #, 
           'allensdk.brain_observatory.ecephys.write_nwb']

data_directory = '/mnt/md0/data'
resort_directory = '/mnt/hdd0/RE-SORT'

df = pd.DataFrame()

for mouse in mice:

    try:
        mouse_directory = data_directory + '/mouse' + str(mouse)

        if os.path.exists(mouse_directory):
            probe_data_directory = resort_directory + '/mouse' + str(mouse)

            pkl_file = glob.glob(mouse_directory + '/*.stim.pkl')[0]
            session_id = os.path.basename(pkl_file).split('.')[0]

            print(session_id)

            print(mouse_directory)

            for module in modules:

                input_json = os.path.join(json_directory, session_id + '-' + module + '-input.json')
                output_json = os.path.join(json_directory, session_id + '-' + module + '-output.json')

                createInputJson(mouse_directory, probe_data_directory, module, input_json)
                
                print('Running ' + module)

                command_string = ["python", "-W", "ignore", "-m", module, 
                                "--input_json", input_json,
                                "--output_json", output_json]

                subprocess.check_call(command_string)
    except:
        print('Error processing')

