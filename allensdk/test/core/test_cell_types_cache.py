# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core.cell_types_cache import ReporterStatus as RS
from allensdk.ephys.feature_extractor import EphysFeatureExtractor
from allensdk.core.swc import Marker
import numpy as np
import pytest
from pandas.core.frame import DataFrame


@pytest.fixture
def cell_id():
    cell_id = 480114344

    return cell_id


@pytest.fixture
def cache_fixture():
    # Instantiate the CellTypesCache instance.  The manifest_file argument
    # tells it where to store the manifest, which is a JSON file that tracks
    # file paths.  If you supply a relative path (like this), it will go
    # into your current working directory
    ctc = CellTypesCache(manifest_file='cell_types/cell_types_manifest.json')

    specimen_id = 464212183

    # this saves the NWB file to 'cell_types/specimen_464212183/ephys.nwb'
    data_set = ctc.get_ephys_data(specimen_id)

    return ctc, data_set


def test_sweep_data(cache_fixture):
    _, data_set = cache_fixture

    sweep_number = 30
    sweep_data = data_set.get_sweep(sweep_number)

    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1] + 1]  # in A
    v = sweep_data["response"][0:index_range[1] + 1]  # in V
    i *= 1e12  # to pA
    v *= 1e3  # to mV

    sampling_rate = sweep_data["sampling_rate"]  # in Hz

    assert sampling_rate == 200000.0
    assert len(sweep_data['stimulus']) == sweep_data['index_range'][1] + 1
    assert len(sweep_data['response']) == sweep_data['index_range'][1] + 1


def test_get_cells_require_morphology(cache_fixture):
    ctc, _ = cache_fixture
    # this downloads metadata for all cells with morphology images
    cells = ctc.get_cells(require_morphology=True)
    assert len(cells) > 0
    print("Cells with morphology images: ", len(cells))


def test_get_cells_require_reconstruction(cache_fixture):
    ctc, _ = cache_fixture
    # cells with reconstructions
    cells = ctc.get_cells(require_reconstruction=True)
    assert len(cells) > 0
    print("Cells with reconstructions: ", len(cells))


def test_get_cells_reporter_positive(cache_fixture):
    ctc, _ = cache_fixture
    # all cre positive cells
    cells = ctc.get_cells(reporter_status=RS.POSITIVE)
    print("Cre-positive cells: ", len(cells))
    assert len(cells) > 0


def test_get_cells_reporter_negative(cache_fixture):
    ctc, _ = cache_fixture

    # cre negative cells with reconstructions
    cells = ctc.get_cells(require_reconstruction=True,
                          reporter_status=RS.NEGATIVE)
    print("Cre-negative cells with reconstructions: ", len(cells))
    assert len(cells) > 0


def test_get_cells_compartment_list(cache_fixture,
                                    cell_id):
    ctc, _ = cache_fixture

    # download and open an SWC file
    morphology = ctc.get_reconstruction(cell_id)

    # the compartment list has all of the nodes in the file
    print(morphology.compartment_list[0])

    # download and open an SWC file
    cell_id = 480114344
    morphology = ctc.get_reconstruction(cell_id)

    # the compartment list has all of the nodes in the file
    print(morphology.compartment_list[0])

    for n in morphology.compartment_list:
        for c in morphology.children_of(n):
            assert 'x' in n
            assert 'y' in n
            assert 'z' in n
            assert 'x' in c
            assert 'y' in c
            assert 'z' in c


def test_get_reconstruction_markers(cache_fixture,
                                    cell_id):
    ctc, _ = cache_fixture
    # download and open a marker file
    markers = ctc.get_reconstruction_markers(cell_id)
    print(len(markers))
    print(markers[0])
    assert len(markers) == 21

    # cut dendrite markers
    dm = [m for m in markers if m['name'] == Marker.CUT_DENDRITE]
    assert len(dm) > 0

    # no reconstruction markers
    nm = [m for m in markers if m['name'] == Marker.NO_RECONSTRUCTION]
    assert len(nm) > 0


def test_cell_types_cache_3(cache_fixture):
    ctc, _ = cache_fixture

    # download all electrophysiology features for all cells
    ephys_features = ctc.get_ephys_features()

    # filter down to a specific cell
    specimen_id = 464212183
    cell_ephys_features = [f for f in ephys_features if f[
        'specimen_id'] == specimen_id]

    updown = np.array([f['upstroke_downstroke_ratio_long_square']
                       for f in ephys_features], dtype=float)
    fasttrough = np.array([f['fast_trough_v_long_square']
                           for f in ephys_features], dtype=float)

    A = np.vstack([fasttrough, np.ones_like(updown)]).T
    print("First 5 rows of A:")
    print(A[:5, :])

    m, c = np.linalg.lstsq(A, updown)[0]
    print("m", m, "c", c)


def test_get_ephys_features(cache_fixture):
    ctc, _ = cache_fixture

    ephys_features = ctc.get_ephys_features()
    cells = ctc.get_cells()

    cell_index = {c['id']: c for c in cells}

    dendrite_types = ['spiny', 'aspiny']
    data = {}

    # group fast trough depth and upstroke downstroke ratio values by cell
    # dendrite type
    for dendrite_type in dendrite_types:
        type_features = [f for f in ephys_features if cell_index[
            f['specimen_id']]['dendrite_type'] == dendrite_type]
        data[dendrite_type] = {
            "fasttrough": [f['fast_trough_v_long_square'] for f in type_features],
            "updown": [f['upstroke_downstroke_ratio_short_square'] for f in type_features],
        }

    assert len(data['spiny']['fasttrough']) > 0
    assert len(data['aspiny']['fasttrough']) > 0
    assert len(data['spiny']['updown']) > 0
    assert len(data['aspiny']['updown']) > 0


def test_cell_types_cache_get_morphology_features(cache_fixture):
    ctc, _ = cache_fixture
    morphology_features = ctc.get_morphology_features()

    assert morphology_features is not None


# download all morphology features for cells with reconstructions
def test_cell_types_cache_get_ephys_sweeps(cache_fixture):
    ctc, _ = cache_fixture
    ephys_sweeps = ctc.get_ephys_sweeps(464212183)

    assert ephys_sweeps is not None


def test_cell_types_all_features_non_dataframe(cache_fixture):
    ctc, _ = cache_fixture
    all_features = ctc.get_all_features(
        dataframe=False, require_reconstruction=True)

    assert all_features is not None


def test_cell_types_cache_feature_extractor(cache_fixture):
    ctc, _ = cache_fixture

    # or download both morphology and ephys features
    # this time we'll ask the cache to return a pandas dataframe
    all_features = ctc.get_all_features(
        dataframe=True, require_reconstruction=True)

    assert isinstance(all_features, DataFrame)


def test_cell_types_get_sweep(cache_fixture):
    _, data_set = cache_fixture

    sweep_number = 35
    sweep_data = data_set.get_sweep(sweep_number)

    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1] + 1]  # in A
    v = sweep_data["response"][0:index_range[1] + 1]  # in V
    i *= 1e12  # to pA
    v *= 1e3  # to mV

    sampling_rate = sweep_data["sampling_rate"]  # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)

    fx = EphysFeatureExtractor()

    stim_start = 1.0
    stim_duration = 1.0

    fx.process_instance("", v, i, t, stim_start, stim_duration, "")
    feature_data = fx.feature_list[0].mean
    print("Avg spike width: {:.2f} ms".format(feature_data['width']))
    print("Avg spike threshold: {:.1f} mV".format(feature_data["threshold"]))

    spike_times = [s["t"] for s in feature_data["spikes"]]

    assert len(spike_times) == 54


def test_build_manifest(cache_fixture):
    ctc, _ = cache_fixture
    ctc.build_manifest('test_manifest.json')

    assert True
