import pytest
import copy
import pathlib
import pandas as pd

from allensdk.test_utilities.custom_comparators import (
    stimulus_pickle_equivalence)

from allensdk.core.pickle_utils import (
    _sanitize_list,
    _sanitize_dict,
    _sanitize_tuple,
    _sanitize_list_or_tuple,
    _sanitize_pickle_data,
    load_and_sanitize_pickle)


@pytest.fixture
def list_data_fixture():
    """
    Return a dict.

    'input' maps to a list that has bytes where we want strings.
    'output' maps to the desired list with bytes cast as strings.
    """

    input_data = [b'happy', 2.3, 5, 'pig', b'banana']
    output_data = ['happy', 2.3, 5, 'pig', 'banana']
    return {'input': input_data, 'output': output_data}


@pytest.fixture
def dict_data_fixture():
    """
    Return a dict.

    'input' maps to a dict that has bytes where we want strings.
    'output' maps to the desired dicts with bytes cast as strings.
    """

    input_data = {b'the_first': 2.1,
                  'the_second': b'funny',
                  b'the_third': 'two',
                  b'the_fourth': b'three'}

    output_data = {'the_first': 2.1,
                   'the_second': 'funny',
                   'the_third': 'two',
                   'the_fourth': 'three'}

    return {'input': input_data, 'output': output_data}


@pytest.fixture
def nested_dict_data_fixture():
    """
    Return a dict.

    'input' maps to a dict that has bytes where we want strings.
    'output' maps to the desired dicts with bytes cast as strings.
    """

    input_data = {
        b'a': [1, b'b', 2, 'c'],
        'c': {b'd': 4, b'e': b'f'},
        b'g': b'h'}

    output_data = {
        'a': [1, 'b', 2, 'c'],
        'c': {'d': 4, 'e': 'f'},
        'g': 'h'}

    return {'input': input_data, 'output': output_data}


@pytest.fixture
def nested_list_data_fixture(
        list_data_fixture,
        nested_dict_data_fixture):
    """
    Return a dict.

    'input' maps to a list that has bytes where we want strings.
    'output' maps to the desired list with bytes cast as strings.
    """
    input_data = [copy.deepcopy(list_data_fixture['input']),
                  copy.deepcopy(nested_dict_data_fixture['input'])]

    output_data = [copy.deepcopy(list_data_fixture['output']),
                   copy.deepcopy(nested_dict_data_fixture['output'])]

    return {'input': input_data, 'output': output_data}


@pytest.mark.parametrize('nested', [True, False])
def test_sanitize_list(
        list_data_fixture,
        nested_list_data_fixture,
        nested):
    """
    Test that _sanitize_list behaves well on an un-nested list
    """
    if nested:
        input_data = copy.deepcopy(nested_list_data_fixture['input'])
        output_data = nested_list_data_fixture['output']
    else:
        input_data = copy.deepcopy(list_data_fixture['input'])
        output_data = list_data_fixture['output']

    for_later = copy.deepcopy(input_data)
    actual = _sanitize_list(input_data)
    assert actual == output_data
    assert not actual == for_later


@pytest.mark.parametrize('nested', [True, False])
def test_sanitize_dict(
        dict_data_fixture,
        nested_dict_data_fixture,
        nested):
    """
    Test that _sanitize_dict behaves well on un-nested dict
    """
    if nested:
        input_data = copy.deepcopy(nested_dict_data_fixture['input'])
        output_data = nested_dict_data_fixture['output']
    else:
        input_data = copy.deepcopy(dict_data_fixture['input'])
        output_data = dict_data_fixture['output']
    for_later = copy.deepcopy(input_data)
    actual = _sanitize_dict(input_data)
    assert actual == output_data
    assert not actual == for_later


def test_sanitize_tuple():
    """
    Test that _sanitize_tuple works as expected
    """
    input_data = (['a', b'b', 'c', 2],
                  'cat',
                  b'dog',
                  {b'd': 2, 'e': 3, b'f': b'g'})

    expected_data = (['a', 'b', 'c', 2],
                     'cat',
                     'dog',
                     {'d': 2, 'e': 3, 'f': 'g'})

    actual = _sanitize_tuple(input_data)
    assert actual == expected_data


def test_sanitize_list_or_tuple():
    """
    Test that _sanitize_list_or_tuple works as expected
    """
    input_data = (['a', b'b', 'c', 2],
                  'cat',
                  b'dog',
                  {b'd': 2, 'e': 3, b'f': b'g'})

    expected_data = (['a', 'b', 'c', 2],
                     'cat',
                     'dog',
                     {'d': 2, 'e': 3, 'f': 'g'})

    actual = _sanitize_list_or_tuple(input_data)
    assert actual == expected_data

    input_data = [['h', b'i', 'j', 2],
                  'frog',
                  b'fly',
                  {b'k': 2, 'l': 3, b'm': b'n'}]

    expected_data = [['h', 'i', 'j', 2],
                     'frog',
                     'fly',
                     {'k': 2, 'l': 3, 'm': 'n'}]
    actual = _sanitize_list_or_tuple(input_data)
    assert actual == expected_data


def test_sanitize_pickle_data(
        nested_list_data_fixture,
        nested_dict_data_fixture):
    """
    Test user-facing sanitization method
    """
    actual = _sanitize_pickle_data(
            copy.deepcopy(nested_list_data_fixture['input']))
    assert actual == nested_list_data_fixture['output']

    actual = _sanitize_pickle_data(
            copy.deepcopy(nested_dict_data_fixture['input']))
    assert actual == nested_dict_data_fixture['output']


def test_load_and_sanitize_error():
    """
    Make sure load_and_sanitize_pickle raises an error if given
    something that is neither .pkl or .gz
    """
    with pytest.raises(ValueError, match='Can open .pkl and .gz'):
        load_and_sanitize_pickle(pickle_path='junk.txt')

    with pytest.raises(ValueError, match='Can open .pkl and .gz'):
        load_and_sanitize_pickle(pickle_path=pathlib.Path('junk.txt'))


def test_local_pickle_equivalence():
    """
    make sure that, when we read in a stimulus pickle file directly and
    sanitize its contents, we get a result equivalent to pd.read_pickle
    """

    this_dir = pathlib.Path(__file__).parent.parent
    pkl_path = this_dir / 'brain_observatory/behavior/resources'
    pkl_path = pkl_path / 'example_stimulus.pkl.gz'

    pd_data = pd.read_pickle(pkl_path)
    sanitized_data = load_and_sanitize_pickle(
                        pickle_path=pkl_path)
    assert stimulus_pickle_equivalence(sanitized_data, pd_data)


@pytest.mark.requires_bamboo
def test_pickle_equivalence():
    """
    make sure that, when we read in a stimulus pickle file directly and
    sanitize its contents, we get a result equivalent to pd.read_pickle
    """

    file_path = pathlib.Path(
        "/allen/programs/braintv/production/visualbehavior/"
        "prod2/specimen_850862430/behavior_session_951520319/951410079.pkl")

    pd_data = pd.read_pickle(file_path)
    sanitized_data = load_and_sanitize_pickle(
                        pickle_path=file_path)
    assert stimulus_pickle_equivalence(sanitized_data, pd_data)
