import pytest
import pandas as pd
import numpy as np
from skimage import draw
from allensdk.internal.brain_observatory import roi_filter
from allensdk.internal.brain_observatory import roi_filter_utils
from allensdk.internal.pipeline_modules import run_roi_filter

OVERLAP_THRESHOLD = 0.9


class TestSegmentation(object):
    def __init__(self, stack, n_rois, has_duplicates, has_unions):
        self.stack = stack
        self.n_rois = n_rois
        self.has_duplicates = has_duplicates
        self.has_unions = has_unions


def create_mask_plane(img_shape, dot_positions, radius=15):
    img = np.zeros(img_shape, dtype=np.uint8)
    for r, c in dot_positions:
        img[draw.circle(r, c, radius, shape=img_shape)] = 1
    return img


@pytest.fixture(params=[(False, False), (True, True),
                        (False, True), (True, False)])
def segmentation(request):
    has_unions, has_duplicates = request.param
    plane1 = create_mask_plane((200,200), [(20,20), (130,60), (170,110)])
    plane2 = create_mask_plane((200,200), [(130,85)])
    n_rois = 4
    masks = [plane1, plane2]

    if has_unions:
        uplane = create_mask_plane((200,200), [(131,62), (131, 85)])
        masks.append(uplane)
        n_rois += 1
    if has_duplicates:
        dplane = create_mask_plane((200,200), [(21,20)])
        masks.append(dplane)
        n_rois += 1

    return TestSegmentation(np.array(masks), n_rois, has_duplicates,
                            has_unions)


@pytest.fixture
def object_list():
    columns = ["index", "traceindex", "tempIndex", "cx", "cy", "mask2Frame",
               "frame", "object", "minx", "miny", "maxx", "maxy", "area",
               "shape0", "shape1", "eXcluded", "meanInt0", "maxInt0",
               "meanInt1", "maxInt1", "maxMeanRatio", "snpoffsetmean",
               "snpoffsetstdv", "act2", "act3", "OvlpCount", "OvlpAreaPer",
               "OvlpObj0", "corcoef0", "OvlpObj1", "corcoef1"]
    data = [
        [0, 0, 112, 363, 12, 0, 81, 1, 354, 5, 371, 18, 170, 0.679, 9, 0, 49,
         73, 32, 54, 0.6875, -18.598810, 12.540119, 2778, 995, 1, 82, 85,
         -1.000, 0, 0.000],
        [1, 1, 12, 224, 13, 0, 2, 1, 218, 8, 230, 18, 106, 0.653, 10, 11, 30,
         62, 12,  23, 0.9167, -34.688274, 16.209919, 2818, 390, 0, 0, 0,
         0.000, 0, 0.000],
        [2, 999, 109, 323, 9, 0, 206, 2, 315, 2, 331, 22, 193, 0.454, 16, 2,
         123, 255, 92, 225, 1.4457, 0.000000, 0.000000, 0, 0, 0, 0, 0, 0.000,
         0, 0.000]
    ]
    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture(scope="module")
def xy_data():
    data = ["0.5,1.7","-1.2,2.5"]
    return data


@pytest.fixture(scope="module")
def old_csv(tmpdir_factory, xy_data):
    data = ["0,{},154.086,-14.9831,0,0,1,0.0254625".format(xy_data[0]),
            "1,{},-0.78758,-2.39286,0,0,0,0.348251".format(xy_data[1])]
    filename = str(tmpdir_factory.mktemp("test").join("old.csv"))
    with open(filename, "w") as f:
        f.write("\n".join(data))
    return filename


@pytest.fixture(scope="module")
def new_csv(tmpdir_factory, xy_data):
    data = ["framenumber,x,y,correlation,input_x,input_y,estimate",
            "0,{},0.65566,3.00745,-1.75258,PhaseCorrelated".format(xy_data[0]),
            "1,{},0.65727,3.15259,-2.8105,PhaseCorrelated".format(xy_data[1])]
    filename = str(tmpdir_factory.mktemp("test").join("new.csv"))
    with open(filename, "w") as f:
        f.write("\n".join(data))
    return filename


def model_data(ol, is_valid=True):
    training_columns = list(ol.columns)
    if is_valid:
        training_columns.extend([1, "depth", "driver1", "driver2", "reporter1"])
    else:
        training_columns.extend([2, "depth", "driver1", "driver2", "reporter1"])
    data = {"structure_ids": [1],
            "drivers": ["driver1", "driver2"],
            "reporters": ["reporter1"],
            "training_features": pd.DataFrame(columns=training_columns)}
    return data


def test_calculate_max_border_all_outliers():
    df = pd.DataFrame(
        np.ones((100,9)),
        columns=["index", "x", "y", "a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError):
        border = roi_filter_utils.calculate_max_border(df, 0)


def test_get_rois(segmentation):
    rois = roi_filter_utils.get_rois(segmentation.stack)
    assert(len(rois) == segmentation.n_rois)


def test_label_unions_and_duplicates(segmentation):
    rois = roi_filter_utils.get_rois(segmentation.stack)
    rois = roi_filter.label_unions_and_duplicates(rois, OVERLAP_THRESHOLD)
    duplicates = False
    unions = False
    for roi in rois:
        if "duplicate" in roi.labels:
            duplicates |= 1
        if "union" in roi.labels:
            unions |= 1
    assert(duplicates == segmentation.has_duplicates)
    assert(unions == segmentation.has_unions)


def test_create_feature_array(object_list):
    depth = 250
    structure_id = 1
    drivers = ["driver1", "driver2"]
    reporters = ["reporter1"]
    passing_data = model_data(object_list, True)
    failing_data = model_data(object_list, False)
    with pytest.raises(KeyError):
        roi_filter.create_feature_array(failing_data, object_list, depth,
                                        structure_id, drivers, reporters)
    feature_array = roi_filter.create_feature_array(passing_data, object_list,
                                                    depth, structure_id,
                                                    drivers, reporters)
    assert(np.all(feature_array.columns ==
           passing_data["training_features"].columns))


def test_training_label_classifier(object_list):
    classifier = roi_filter_utils.TrainingMultiLabelClassifier()
    assert(classifier.labels == sorted(roi_filter_utils.CRITERIA().keys()))


def test_read_csv(old_csv, new_csv):
    assert(not run_roi_filter.is_deprecated_motion_file(new_csv))
    assert(run_roi_filter.is_deprecated_motion_file(old_csv))
    old_data = run_roi_filter.load_rigid_motion_transform(old_csv)
    new_data = run_roi_filter.load_rigid_motion_transform(new_csv)
    assert(np.all(np.isclose(old_data["x"], new_data["x"])))
    assert(np.all(np.isclose(old_data["y"], new_data["y"])))
