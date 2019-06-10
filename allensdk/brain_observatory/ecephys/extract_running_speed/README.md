Extract running speed
=====================
Calculates an average running speed for the mouse on each stimulus frame.


Running
-------
```
python -m ecephys_pipeline.modules.extract_running_speed --input_json <path to input json> --output_json <path to output json>
```
See the schema file for detailed information about input json contents.


Input data
----------
- Stimulus pickle : Written by camstim (http://aibspi/braintv/camstim). Contains information about the stimuli that were 
presented in this experiment.
- Sync h5 : Contains information about the times at which each frame was presented.


Output data
-----------
- Running speeds npy : for each frame, the average running speed on that frame
- Timestamps npy : for each frame, the onset time of that frame (s, master clock)