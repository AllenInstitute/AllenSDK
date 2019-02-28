import json
import pandas as pd
import numpy as np

def get_input_extract_traces_json(input_extract_traces_file):
    # processed_dir = get_processed_dir(lims_data)
    # json_file = [file for file in os.listdir(processed_dir) if 'input_extract_traces.json' in file]
    # json_path = lims_data.curr_input_extract_traces_file #os.path.join(processed_dir, json_file[0])
    with open(input_extract_traces_file, 'r') as w:
        jin = json.load(w)
    return jin

def parse_mask_string(mask_string):
    # convert ruby json array ouput to python 2D array
    # needed for segmentation output prior to 10/10/17 due to change in how masks were saved
    mask = []
    row_length = -1
    for i in range(1, len(mask_string) - 1):
        c = mask_string[i]
        if c == '{':
            row = []
        elif c == '}':
            mask.append(row)
            if row_length < 1:
                row_length = len(row)
        elif c == 'f':
            row.append(False)
        elif c == 't':
            row.append(True)
    return np.asarray(mask)

def add_cell_roi_ids_to_roi_metrics(roi_metrics, roi_locations):
    # add roi ids to objectlist
    ids = []
    for row in roi_metrics.index:
        minx = roi_metrics.iloc[row][' minx']
        miny = roi_metrics.iloc[row][' miny']
        id = roi_locations[(roi_locations.x == minx) & (roi_locations.y == miny)].id.values[0]
        ids.append(id)
    roi_metrics['cell_roi_id'] = ids
    return roi_metrics

def get_roi_locations(input_extract_traces_file):
    jin = get_input_extract_traces_json(input_extract_traces_file)
    h = jin["image"]["height"]
    w = jin["image"]["width"]
    rois = jin["rois"]
    # get data out of json and into dataframe
    roi_locations_list = []
    for i in range(len(rois)):
        roi = rois[i]
        if roi['mask'][0] == '{':
            mask = parse_mask_string(roi['mask'])
        else:
            mask = roi["mask"]
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[int(roi["y"]):int(roi["y"]) + int(roi["height"]), int(roi["x"]):int(roi["x"]) + int(roi["width"])] = mask

        roi_locations_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], binary_mask])
    roi_locations = pd.DataFrame(data=roi_locations_list, columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'])
    return roi_locations

def get_roi_metrics(input_extract_traces_file, ophys_experiment_id, objectlist_file):
    # objectlist.txt contains metrics associated with segmentation masks
    roi_metrics = pd.read_csv(objectlist_file)
    # get roi_locations and add unfiltered cell index
    roi_locations = get_roi_locations(input_extract_traces_file)
    roi_names = np.sort(roi_locations.id.values)
    roi_locations['unfiltered_cell_index'] = [np.where(roi_names == id)[0][0] for id in roi_locations.id.values]
    # add cell ids to roi_metrics from roi_locations
    roi_metrics = add_cell_roi_ids_to_roi_metrics(roi_metrics, roi_locations)
    # merge roi_metrics and roi_locations
    roi_metrics['id'] = roi_metrics.cell_roi_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on='id')
    unfiltered_roi_metrics = roi_metrics
    # remove invalid roi_metrics
    roi_metrics = roi_metrics[roi_metrics.valid == True]
    # hack for expt 692342909 with 2 rois at same location - need a long term solution for this!
    if ophys_experiment_id == 692342909:
        # logger.info('removing bad cell')
        roi_metrics = roi_metrics[roi_metrics.cell_roi_id.isin([692357032, 692356966]) == False]
    # hack to get rid of cases with 2 rois at the same location
    for cell_roi_id in roi_metrics.cell_roi_id.values:
        roi_data = roi_metrics[roi_metrics.cell_roi_id == cell_roi_id]
        if len(roi_data) > 1:
            ind = roi_data.index
            roi_metrics = roi_metrics.drop(index=ind.values)
    # add filtered cell index
    cell_index = [np.where(np.sort(roi_metrics.cell_roi_id.values) == id)[0][0] for id in
                  roi_metrics.cell_roi_id.values]
    roi_metrics['cell_index'] = cell_index
    return {'filtered':roi_metrics, 'unfiltered':unfiltered_roi_metrics}
