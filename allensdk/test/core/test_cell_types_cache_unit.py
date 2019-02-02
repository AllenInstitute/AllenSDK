# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2016-2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import allensdk.core.cell_types_cache as CTC
from allensdk.core.cell_types_cache import ReporterStatus as RS
import pytest
from pandas.core.frame import DataFrame
from allensdk.config import enable_console_log
from mock import MagicMock, patch, call, mock_open
from six.moves import builtins
import itertools as it
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
import os

_MOCK_PATH = '/path/to/xyz.txt'


@pytest.fixture(scope="session", autouse=True)
def console_log():
    enable_console_log()


@pytest.fixture
def cell_id():
    cell_id = 480114344

    return cell_id


@pytest.fixture
def cached_csv(tmpdir_factory):
    csv = str(tmpdir_factory.mktemp("cache_test").join("data.csv"))
    return csv


@pytest.fixture
def cache_fixture(tmpdir_factory):
    # Instantiate the CellTypesCache instance.  The manifest_file argument
    # tells it where to store the manifest, which is a JSON file that tracks
    # file paths.  If you supply a relative path, it will go
    # into your current working directory
    manifest_file = str(tmpdir_factory.mktemp("ctc").join("manifest.json"))
    ctc = CTC.CellTypesCache(manifest_file=manifest_file)

    return ctc


@pytest.mark.parametrize('path_exists',
                         (False, True))
@patch('allensdk.core.cell_types_cache.NwbDataSet')
def test_sweep_data_with_api(mock_nwb,
                             cache_fixture,
                             path_exists):
    ctc = cache_fixture

    specimen_id = 464212183

    ephys_result = [{'ephys_result':
                     {'well_known_files': [
                      {'download_link': '/path/to/data.nwb' }]}}]

    # this saves the NWB file to 'cell_types/specimen_464212183/ephys.nwb'
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query',
                            return_value=ephys_result)) as query_mock:
                with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                    with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs') as mkd:
                        mock_nwb.reset_mock()
                        _ = ctc.get_ephys_data(specimen_id, _MOCK_PATH)

    assert ope.called
    if path_exists:
        mock_nwb.assert_called_once_with(_MOCK_PATH)
        assert not query_mock.called
        assert not mkd.called
    else:
        # both levels of cacheable methods check if the directory exists.
        assert mkd.call_args_list == [call(_MOCK_PATH)]
        assert query_mock.called
        mock_http.assert_called_once_with('http://api.brain-map.org/path/to/data.nwb',
                                          _MOCK_PATH)


def test_sweep_data_exception(cache_fixture):
    ctc = cache_fixture

    specimen_id = 464212183

    ephys_result = [{'ephys_result':
                     {'well_known_files': [] }}]

    with pytest.raises(Exception) as exc:
        with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
                with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                    MagicMock(name='model query',
                                return_value=ephys_result)) as query_mock:
                    with patch('os.path.exists', MagicMock(return_value=False)) as ope:
                        with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                            with patch('allensdk.core.cell_types_cache.NwbDataSet') as nwb:
                                _ = ctc.get_ephys_data(specimen_id)
    
    assert 'has no ephys data' in str(exc.value)


@pytest.mark.parametrize('path_exists,morph_flag,recon_flag,statuses,species,simple',
                         it.product((False, True),
                                    (False, True),
                                    (False, True),
                                    (RS.POSITIVE, ['list', 'of', 'statuses']),
                                    (None, ['mouse'], ['human']),
                                    (False,)))

def test_get_cells(cache_fixture,
                   path_exists,
                   morph_flag,
                   recon_flag,
                   statuses,
                   species,
                   simple):
    ctc = cache_fixture
    # this downloads metadata for all cells with morphology images
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
            with patch('allensdk.core.json_utilities.read',
                       return_value=['mock_cells_from_server']) as ju_read:
                with patch('allensdk.api.queries.cell_types_api.CellTypesApi.list_cells_api',
                           MagicMock(return_value=['mock_cells_from_server'])) as list_cells_mock:
                    with patch('allensdk.api.queries.cell_types_api.CellTypesApi.filter_cells_api',
                               MagicMock(return_value=['mock_cells'])) as filter_cells_mock:
                        with patch('allensdk.core.json_utilities.write') as ju_write:
                            cells = ctc.get_cells(require_morphology=morph_flag,
                                                  require_reconstruction=recon_flag,
                                                  reporter_status=statuses,
                                                  species=species,
                                                  simple=simple)

    assert cells == ['mock_cells']

    if (statuses == RS.POSITIVE):
        expected_status = [statuses]
    else:
        expected_status = statuses

    filter_cells_mock.assert_called_once_with(['mock_cells_from_server'],
                                              morph_flag,
                                              recon_flag,
                                              expected_status,
                                              species,
                                              simple)


@pytest.mark.parametrize('path_exists,morph_flag,recon_flag,statuses',
                         it.product((False, True),
                                    (False, True),
                                    (False, True),
                                    (RS.POSITIVE, ['list', 'of', 'statuses'])))
def test_get_cells_with_api(cache_fixture,
                            path_exists,
                            morph_flag,
                            recon_flag,
                            statuses):
    ctc = cache_fixture

    # note, this is only a mock for coverage,
    # and has not a lot of relation to the actual data form
    sweeps = [1, 2, 3]
    return_dicts = [{'sweep_number': x,
                     'tags': ['what - ever'],
                     'neuron_reconstructions' : [],
                     'data_sets': [],
                     'reporter_status': 'whatever',
                     'has_morphology': False,
                     'has_reconstruction': False,
                     'donor': { 'transgenic_lines': [{'transgenic_line_type_name': 'driver',
                                                      'name': 'harold'}]},
                     'cell_reporter': {'name': 'tired'},
                     'specimen_tags': [{'name': 'a - b',
                                        'value': 123}]} for x \
                     in sweeps]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query', return_value=return_dicts)) as query_mock:
            with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                with patch('allensdk.core.json_utilities.read',
                        return_value=return_dicts) as ju_read:
                    with patch('allensdk.core.json_utilities.write') as ju_write:
                        with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                            cells = ctc.get_cells(require_morphology=morph_flag,
                                                  require_reconstruction=recon_flag,
                                                  reporter_status=statuses,
                                                  simple=True)
    if path_exists:
        ju_read.assert_called_once_with(_MOCK_PATH)
    else:
        assert ju_write.called

@pytest.mark.parametrize('path_exists',
                         (False, True))
def test_get_reconstruction(cache_fixture,
                            cell_id,
                            path_exists):
    ctc = cache_fixture

    save_recon = \
        'allensdk.api.queries.cell_types_api.CellTypesApi.save_reconstruction'
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch(save_recon) as save_recon_mock:
            with patch('allensdk.core.swc.read_swc') as read_swc_mock:
                # download and open an SWC file
                _ = ctc.get_reconstruction(cell_id)

    if path_exists is False:
        save_recon_mock.assert_called_once_with(cell_id,
                                                _MOCK_PATH)

    read_swc_mock.assert_called_once_with(_MOCK_PATH)


@pytest.mark.parametrize('path_exists',
                         (False, True))
@patch.object(DataFrame, "to_csv")
def test_get_reconstruction_with_api(to_csv,
                                     cache_fixture,
                                     cell_id,
                                     path_exists):
    ctc = cache_fixture

    reconstruction_data = [{'neuron_reconstructions': [
                            {'well_known_files': [
                             {'download_link': 'http://example.org'}]}]}]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                    MagicMock(name='model query',
                                return_value=reconstruction_data)) as query_mock:
                with patch('allensdk.core.swc.read_swc') as read_swc_mock:
                    with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                        with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                            _ = ctc.get_reconstruction(cell_id)

    if path_exists:
        read_swc_mock.assert_called_once_with(_MOCK_PATH)
    else:
        assert query_mock.called


@patch.object(DataFrame, "to_csv")
def test_get_reconstruction_exception(to_csv,
                                      cache_fixture,
                                      cell_id):
    ctc = cache_fixture

    reconstruction_data = [{'neuron_reconstructions': [
                            {'well_known_files': None}]}]

    with pytest.raises(Exception) as exc:
        with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
                with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                        MagicMock(name='model query',
                                    return_value=reconstruction_data)) as query_mock:
                    with patch('allensdk.core.swc.read_swc') as read_swc_mock:
                        with patch('os.path.exists', MagicMock(return_value=False)) as ope:
                            with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                                _ = ctc.get_reconstruction(cell_id)

    assert 'has no reconstruction' in str(exc.value)


@pytest.mark.parametrize('path_exists,lookup_error',
                         it.product((False, True),
                                    (False, True)))
def test_get_reconstruction_markers(cache_fixture,
                                    cell_id,
                                    path_exists,
                                    lookup_error):
    ctc = cache_fixture

    if lookup_error:
        def lookup(i, n):
            raise(LookupError('mock lookup error'))
    else:
        def lookup(i, n):
            return

    save_recon_marker = \
        'allensdk.api.queries.cell_types_api.CellTypesApi.save_reconstruction_markers'

    # download and open a marker file
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch(save_recon_marker,
                MagicMock(side_effect=lookup)) as save_recon_markers_mock:
            with patch('allensdk.core.swc.read_marker_file') as read_marker_mock:
                _ = ctc.get_reconstruction_markers(cell_id)

    if path_exists is False:
        save_recon_markers_mock.assert_called_once_with(cell_id,
                                                       _MOCK_PATH)

    if lookup_error:
        assert not read_marker_mock.called
    else:
        read_marker_mock.assert_called_once_with(_MOCK_PATH)


@pytest.mark.parametrize('path_exists,lookup_error',
                         it.product((False, True),
                                    (False, True)))
def test_get_reconstruction_markers_with_api(cache_fixture,
                                             cell_id,
                                             path_exists,
                                             lookup_error):
    ctc = cache_fixture

    reconstruction_data = [{'neuron_reconstructions': [
                            {'well_known_files': [
                             {'download_link': '/mock/path_to_file'}]}]}]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                    MagicMock(name='model query',
                                return_value=reconstruction_data)) as query_mock:
                with patch('allensdk.core.swc.read_marker_file') as marker_mock:
                    with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                        with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                            _ = ctc.get_reconstruction_markers(cell_id)

    if path_exists:
        assert marker_mock.called
    else:
        mock_http.assert_called_once_with('http://api.brain-map.org/mock/path_to_file',
                                          _MOCK_PATH)


def test_get_reconstruction_markers_exception(cache_fixture,
                                              cell_id):
    ctc = cache_fixture

    reconstruction_data = [{'neuron_reconstructions': [
                            {'well_known_files': []}]}]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.retrieve_file_over_http') as mock_http:
            with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                    MagicMock(name='model query',
                                return_value=reconstruction_data)) as query_mock:
                with patch('allensdk.core.swc.read_marker_file') as marker_mock:
                    with patch('os.path.exists', MagicMock(return_value=False)) as ope:
                        with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                            markers = ctc.get_reconstruction_markers(cell_id)

                        assert len(markers) == 0


@pytest.mark.parametrize('dataframe',
                         (False, True))
def test_get_ephys_features(cache_fixture,
                            dataframe):
    ctc = cache_fixture

    api_get_ephys_features = \
        'allensdk.api.queries.cell_types_api.CellTypesApi.get_ephys_features'

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch(api_get_ephys_features) as api_get_ephys_features_mock:
            # download all electrophysiology features for all cells
            _ = ctc.get_ephys_features(dataframe=dataframe)

    assert api_get_ephys_features_mock.called


@pytest.mark.parametrize('df,path_exists',
                         it.product((False,True),
                                    (False,True)))
@patch.object(DataFrame, "to_csv")
@patch("pandas.read_csv")
def test_get_ephys_features_with_api(read_csv,
                                     to_csv,
                                     cache_fixture,
                                     df,
                                     path_exists):
    ctc = cache_fixture

    mock_data = [{'lorem': 1,
                  'ipsum': 2 },
                 {'lorem': 3,
                  'ipsum': 4 }]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query',
                            return_value=mock_data)) as query_mock:
            with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                with patch(builtins.__name__ + '.open',
                        mock_open(),
                        create=True) as open_mock:
                    with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs') as mkd:
                        _ = ctc.get_ephys_features(dataframe=df)

    if path_exists:
        read_csv.assert_called_once_with(_MOCK_PATH, parse_dates=True)
    else:
        mkd.assert_called_once_with(_MOCK_PATH)
        assert query_mock.called


@pytest.mark.parametrize('df', (False, True))
def test_get_ephys_features_cache_roundtrip(cached_csv,
                                            cache_fixture,
                                            df):
    ctc = cache_fixture

    mock_data = [{'lorem': 1,
                  'ipsum': 2 },
                 {'lorem': 3,
                  'ipsum': 4 }]

    with patch.object(ctc, "get_cache_path", return_value=cached_csv):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query',
                            return_value=mock_data)) as query_mock:
            data = ctc.get_ephys_features()
    pandas_data = pd.read_csv(cached_csv, parse_dates=True)

    assert len(data) == 2
    assert sorted(data[0].keys()) == sorted(pandas_data.columns)


@pytest.mark.parametrize('path_exists,df',
                         it.product((False, True),
                                    (False, True)))
@patch.object(DataFrame, "to_csv")
@patch("pandas.read_csv",
              return_value=DataFrame([{ 'stuff': 'whatever'},
                                      { 'stuff': 'nonsense'}]))
def test_get_morphology_features(read_csv,
                                 to_csv,
                                 cache_fixture,
                                 path_exists,
                                 df):
    ctc = cache_fixture

    json_data = [{ 'stuff': 'whatever'},
                 { 'stuff': 'nonsense'}]
    
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
            with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs') as mkd:
                with patch(builtins.__name__ + '.open',
                        mock_open(),
                        create=True) as open_mock:
                    with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                            MagicMock(name='model query',
                                        return_value=json_data)) as query_mock:
                        data = ctc.get_morphology_features(df, _MOCK_PATH)

    if df:
        assert ('stuff' in data) == True
    else:
        assert all(['stuff' in f for f in data])

    
    if path_exists:
        if df:
            read_csv.assert_called_once_with(_MOCK_PATH, parse_dates=True)
        else:
            assert True
        assert not mkd.called
    else:
        assert query_mock.called
        assert mkd.called


@pytest.mark.parametrize('path_exists',
                         (False, True))
def test_get_ephys_sweeps(cache_fixture,
                          path_exists):
    ctc = cache_fixture

    cell_id = 464212183

    get_ephys_sweeps = \
        'allensdk.api.queries.cell_types_api.CellTypesApi.get_ephys_sweeps'
    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch(get_ephys_sweeps) as get_ephys_sweeps_mock:
            with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                with patch('allensdk.core.json_utilities.read',
                        return_value=['mock_data']) as ju_read:
                    with patch('allensdk.core.json_utilities.write') as ju_write:
                        _ = ctc.get_ephys_sweeps(cell_id)

    if path_exists:
        assert ju_read.called_once_with(_MOCK_PATH)
    else:
        assert get_ephys_sweeps_mock.called_once_with(cell_id)


@pytest.mark.parametrize('path_exists',
                         (False, True))
def test_get_ephys_sweeps_with_api(cache_fixture,
                                   path_exists):
    ctc = cache_fixture

    cell_id = 464212183
    sweeps = [1, 2, 3]
    return_dicts = [{'sweep_number': x} for x in sweeps]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query',
                            return_value=return_dicts)) as query_mock:
            with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                    with patch('allensdk.core.json_utilities.read',
                            return_value=['mock_data']) as ju_read:
                        with patch('allensdk.core.json_utilities.write') as ju_write:
                            _ = ctc.get_ephys_sweeps(cell_id)

    # read will be called regardless
    assert ju_read.called_once_with(_MOCK_PATH)

    if path_exists:
        assert not query_mock.called
    else:
        assert query_mock.called


@pytest.mark.parametrize('path_exists,require_reconstruction',
                         it.product((False, True),
                                    (False, True)))
@patch('pandas.DataFrame.merge')
@patch.object(DataFrame, "to_csv")
@patch("pandas.read_csv",
              return_value=DataFrame([{ 'stuff': 'whatever'},
                                      { 'stuff': 'nonsense'}]))
def test_get_all_features(read_csv,
                          to_csv,
                          mock_merge,
                          cache_fixture,
                          path_exists,
                          require_reconstruction):
    ctc = cache_fixture

    sweeps = [1, 2, 3]
    return_dicts = [{'sweep_number': x,
                     'tags': 'whatever'} for x in sweeps]

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.api.queries.cell_types_api.CellTypesApi.model_query',
                MagicMock(name='model query',
                            return_value=return_dicts)) as query_mock:
                with patch('os.path.exists', MagicMock(return_value=path_exists)) as ope:
                    with patch('allensdk.config.manifest.Manifest.safe_make_parent_dirs'):
                        with patch('allensdk.core.json_utilities.read',
                                return_value=return_dicts) as ju_read:
                            with patch(builtins.__name__ + '.open',
                                    mock_open(),
                                    create=True) as open_mock:
                                with patch('allensdk.core.json_utilities.write') as ju_write:
                                    _ = ctc.get_all_features(
                                        require_reconstruction=require_reconstruction)

    if path_exists:
        assert read_csv.called
    else:
        assert query_mock.called
    
    assert mock_merge.called


def test_build_manifest(cache_fixture):
    ctc = cache_fixture
    
    mb_mock = MagicMock(name='manifest builder')

    with patch.object(ctc, "get_cache_path", return_value=_MOCK_PATH):
        with patch('allensdk.core.cell_types_cache.ManifestBuilder',
                return_value=mb_mock):
            ctc.build_manifest('test_manifest.json')

    assert mb_mock.add_path.call_count == 8
    mb_mock.write_json_file.assert_called_once_with('test_manifest.json')
