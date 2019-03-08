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