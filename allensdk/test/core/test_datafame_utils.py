import pytest
import pandas as pd

from allensdk.core.dataframe_utils import (
    patch_df_from_other)


@pytest.fixture
def target_df_fixture():
    """
    Return an example target dataframe with columns 'a', 'b', 'c', 'd'
    """
    data = [
        {'a': 1, 'b': 3.4, 'c': 'apple', 'd': None},
        {'a': 9, 'b': 4.5, 'c': 'banana', 'd': 4.6},
        {'a': 12, 'b': 7.8, 'c': 'pineapple', 'd': 'purple'},
        {'a': 17, 'b': None, 'c': 'papaya', 'd': 11}
    ]
    return pd.DataFrame(data=data)


@pytest.fixture
def source_df_fixture():
    """
    Return an example source dataframe with columns 'a', 'b', 'd', 'e'
    """
    data = [
        {'a': 1, 'b': 7.5, 'd': 'tree', 'e': 'frog'},
        {'a': 12, 'b': 88.9, 'd': None, 'e': 'dog'},
        {'a': 17, 'b': 35.2, 'd': 17, 'e': 'cat'}
    ]
    return pd.DataFrame(data=data)


def test_patch_error_on_no_index(
        target_df_fixture,
        source_df_fixture):
    """
    Test than an exception is raise when you specify an
    index_column that does not exist in one or the other
    dataframe
    """
    with pytest.raises(ValueError, match='not in target_df'):
        patch_df_from_other(
            source_df=source_df_fixture,
            target_df=target_df_fixture,
            index_column='e',
            columns_to_patch=['a', 'b'])

    with pytest.raises(ValueError, match='not in source_df'):
        patch_df_from_other(
            source_df=source_df_fixture,
            target_df=target_df_fixture,
            index_column='c',
            columns_to_patch=['a', 'b'])


def test_patch_error_on_missing_col(
        source_df_fixture,
        target_df_fixture):
    """
    Test that an error is raised when you specify a
    column_to_patch that is not in the source_df
    """
    with pytest.raises(ValueError, match='not in source_df'):
        patch_df_from_other(
            source_df=source_df_fixture,
            target_df=target_df_fixture,
            index_column='a',
            columns_to_patch=['c'])


def test_error_on_not_unique_index(
        target_df_fixture):
    """
    Test that an exception is raised when the values of index_column
    in source_df are not unique
    """
    data = [
        {'a': 1, 'b': 7.5, 'e': 'frog'},
        {'a': 12, 'b': 88.9, 'e': 'dog'},
        {'a': 17, 'b': 35.2, 'e': 'cat'},
        {'a': 17, 'b': 55.7, 'e': 'mosquito'}
    ]
    source_df = pd.DataFrame(data=data)
    with pytest.raises(ValueError, match='in source_df are not unique'):
        patch_df_from_other(
            source_df=source_df,
            target_df=target_df_fixture,
            index_column='a',
            columns_to_patch=['c'])


@pytest.mark.parametrize('original_index', [None, 'a', 'c', 'b'])
def test_patch_no_duplicates(
        source_df_fixture,
        target_df_fixture,
        original_index):
    """
    Test that we get the expected dataframe back in the case
    where there are no duplicate values of index_column
    in target_df

    original_index sets the column on which the target_df
    is originally indexed (since getting the right index on
    output is non-trivial)
    """

    expected_data = [
        {'a': 1, 'b': 7.5, 'c': 'apple', 'd': 'tree'},
        {'a': 9, 'b': 4.5, 'c': 'banana', 'd': 4.6},
        {'a': 12, 'b': 88.9, 'c': 'pineapple', 'd': 'purple'},
        {'a': 17, 'b': 35.2, 'c': 'papaya', 'd': 17}
    ]
    expected_df = pd.DataFrame(data=expected_data)

    if original_index is not None:
        expected_df = expected_df.set_index(original_index)
        target_df = target_df_fixture.copy(deep=True)
        target_df = target_df.set_index(original_index)
    else:
        target_df = target_df_fixture.copy(deep=True)

    actual = patch_df_from_other(
                source_df=source_df_fixture.copy(deep=True),
                target_df=target_df,
                index_column='a',
                columns_to_patch=['b', 'd'])

    pd.testing.assert_frame_equal(actual, expected_df)


@pytest.mark.parametrize('original_index', [None, 'c', 'b'])
def test_patch_with_duplicates(
        source_df_fixture,
        target_df_fixture,
        original_index):
    """
    Test that we get the expected dataframe back in the case
    where there are duplicate values of index_column
    in target_df

    original_index sets the column on which the target_df
    is originally indexed (since getting the right index on
    output is non-trivial)
    """

    data = [
        {'a': 1, 'b': 3.4, 'c': 'apple', 'd': None},
        {'a': 9, 'b': 4.5, 'c': 'banana', 'd': 4.6},
        {'a': 17, 'b': 11.3, 'c': 'tomato', 'd': 'blue'},
        {'a': 12, 'b': 7.8, 'c': 'pineapple', 'd': 'purple'},
        {'a': 17, 'b': None, 'c': 'papaya', 'd': 11}
    ]
    target_df = pd.DataFrame(data=data)

    expected_data = [
        {'a': 1, 'b': 7.5, 'c': 'apple', 'd': 'tree'},
        {'a': 9, 'b': 4.5, 'c': 'banana', 'd': 4.6},
        {'a': 17, 'b': 35.2, 'c': 'tomato', 'd': 17},
        {'a': 12, 'b': 88.9, 'c': 'pineapple', 'd': 'purple'},
        {'a': 17, 'b': 35.2, 'c': 'papaya', 'd': 17}
    ]
    expected_df = pd.DataFrame(data=expected_data)

    if original_index is not None:
        expected_df = expected_df.set_index(original_index)
        target_df = target_df.set_index(original_index)

    actual = patch_df_from_other(
                source_df=source_df_fixture.copy(deep=True),
                target_df=target_df,
                index_column='a',
                columns_to_patch=['b', 'd'])

    pd.testing.assert_frame_equal(actual, expected_df)


def test_patch_new_column(
        target_df_fixture,
        source_df_fixture):
    """
    Test case where we use patch_df_from_other in the case where we are
    adding a column to target_df
    """
    expected_data = [
        {'a': 1, 'b': 3.4, 'c': 'apple', 'd': None, 'e': 'frog'},
        {'a': 9, 'b': 4.5, 'c': 'banana', 'd': 4.6, 'e': None},
        {'a': 12, 'b': 7.8, 'c': 'pineapple', 'd': 'purple', 'e': 'dog'},
        {'a': 17, 'b': None, 'c': 'papaya', 'd': 11, 'e': 'cat'}
    ]
    expected_df = pd.DataFrame(data=expected_data)

    actual = patch_df_from_other(
                source_df=source_df_fixture.copy(deep=True),
                target_df=target_df_fixture,
                index_column='a',
                columns_to_patch=['e'])

    pd.testing.assert_frame_equal(actual, expected_df)
