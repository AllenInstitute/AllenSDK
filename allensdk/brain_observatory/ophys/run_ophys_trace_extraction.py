import sys
import os
from allensdk.internal.core.lims_pipeline_module import run_module, PipelineModule
import allensdk.internal.core.lims_utilities as lu
import h5py
import allensdk.brain_observatory.roi_masks as roi_masks
import numpy as np
import logging


def create_roi_masks(rois, w, h, motion_border):
    roi_list = []

    for roi in rois:
        mask = np.array(roi["mask"], dtype=bool)
        px = np.argwhere(mask)
        px[:,0] += roi["y"]
        px[:,1] += roi["x"]

        mask = roi_masks.create_roi_mask(w, h, motion_border, 
                                         pix_list=px[:,[1,0]], 
                                         label=str(roi["id"]), 
                                         mask_group=roi.get("mask_page",-1))

        roi_list.append(mask)

    # sort by roi id
    roi_list.sort(key=lambda x: x.label)

    return roi_list

def main():
    mod = PipelineModule()
    jin = mod.input_data()

    # find width and height of movie
    stack_file = jin["motion_corrected_stack"]
    with h5py.File(stack_file, "r") as f:
        d = f["data"]
        h = d.shape[1]
        w = d.shape[2]

    # where to store output
    output_directory = jin["storage_directory"]

    # motion border
    border = [jin["motion_border"]["x0"],
              jin["motion_border"]["x1"],
              jin["motion_border"]["y0"],
              jin["motion_border"]["y1"]]

    # create roi mask objects
    roi_mask_list = create_roi_masks(jin["rois"], w, h, border)
    roi_names = [ roi.label for roi in roi_mask_list ]

    # extract traces
    roi_traces, neuropil_traces = roi_masks.calculate_roi_and_neuropil_traces(jin["motion_corrected_stack"], roi_mask_list, border)

    # write ROI traces
    roi_file = os.path.join(output_directory, "roi_traces.h5")
    logging.debug("Writing " + roi_file)
    rf = h5py.File(roi_file, 'w')
    rf["data"] = roi_traces
    rf.create_dataset("roi_names", data=roi_names)
    rf.close()

    # write neuropil traces
    np_file = os.path.join(output_directory, "neuropil_traces.h5")
    logging.debug("Writing " + np_file)
    neup = h5py.File(np_file, 'w')
    neup["data"] = neuropil_traces
    neup.create_dataset("roi_names", data=roi_names)
    neup.close()

    # output json 
    jout = {}
    jout["neuropil_trace_file"] = np_file
    jout["roi_trace_file"] = roi_file

    mod.write_output_data(jout)
   

def get_input_data(experiment_id, output_dir):
    
    sd = lu.query("select storage_directory from ophys_experiments where id = %d" % experiment_id)[0]['storage_directory']
    rois = lu.query("select * from cell_rois where ophys_experiment_id = %d" % experiment_id)

    motion_border = None
    nrois = []
    for roi in rois:
        if motion_border is None:
            motion_border=dict(y1=roi['max_correction_up'],
                               y0=roi['max_correction_down'],
                               x0=roi['max_correction_left'],
                               x1=roi['max_correction_right'])
        nrois.append({
                'width': roi['width'],
                'height': roi['height'], 
                'x': roi['x'],
                'y': roi['y'],
                'id': roi['id'],
                'valid': roi['valid_roi'],
                'mask': roi['mask_matrix']
                })

    movie_path = os.path.join(sd, "processed", "concat_31Hz_0.h5")
    with h5py.File(movie_path, "r") as f:
        s = f["data"].shape
        movie_height = s[1]
        movie_width = s[2]
    

    return dict(
        image=dict(width=movie_width, height=movie_height),
        motion_border=motion_border,
        motion_corrected_stack=movie_path,
        storage_directory=output_dir,
        rois=nrois
        )

def debug(experiment_id, local=False):
    OUTPUT_DIRECTORY = "/data/informatics/CAM/traces"
    PYTHON = "/shared/utils.x86_64/python-2.7/bin/python"
    SDK_PATH = "/data/informatics/CAM/traces/allensdk" 
    SCRIPT = "/data/informatics/CAM/traces/allensdk/allensdk/internal/pipeline_modules/run_ophys_trace_extraction.py"

    exp_dir = os.path.join(OUTPUT_DIRECTORY, str(experiment_id))
    input_data = get_input_data(experiment_id, exp_dir)

    run_module(SCRIPT, 
               input_data, 
               exp_dir, 
               sdk_path=SDK_PATH,
               pbs=dict(vmem=16,
                        job_name="extraces_%d"% experiment_id,
                        walltime="10:00:00"),
               local=local)

if __name__ == "__main__": main()
    
    
