Current Source Density
======================
Computes the current source density for one or more probes


Running
-------
You can run this module from an input json:
```
python -m allensdk.brain_observatory.ecephys.current_source_density --input_json <path to input json> --output_json <path to output json>
```
If you are on the Allen Institute's network, you can also run the module from information in our LIMS:
```
python -m allensdk.brain_observatory.ecephys.current_source_density --source lims --session_id <id of an ecephys_session> --output_root <path to output data directory> --output_json <path to output json>
```


Input data
----------
This module takes in an array of LFP samples and their associated timestamps for each probe. The CSD is calculated for some window in time around the onset of a stimulus. The window, as well as the stimulus, need to be specified. The stimulus table is required in order to determine when these stimulus onsets occured. See the schema file for detailed informtaion on the module inputs.


Output data
-----------
- a (channels X samples) npy file containing CSD data
- a 1D npy file defining sample timestamps in seconds relative to stimulus onset 
- in the output json, an array of channel ids identifying the channels at each row in the csd data file
