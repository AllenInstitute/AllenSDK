from unittest.mock import patch

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .ophys_mixin import \
    OphysMixin


class TestOhysMixin:
    def test__merge_column_values_nothing_to_merge(self):
        """when nothing to merge, df stays same"""
        df = pd.DataFrame({
            'date_of_acquisition': [1, 2],
            'foo': [3, 4]
        })

        with patch.object(OphysMixin, '__init__') as ophys_mixin:
            ophys_mixin.return_value = None
            mixin = OphysMixin()
        mixin._df = df
        mixin._merge_columns()
        pd.testing.assert_frame_equal(df, mixin._df)

    def test__merge_column_values(self):
        """ophys value chosen"""
        df = pd.DataFrame({
            'date_of_acquisition_behavior': [1, 2],
            'date_of_acquisition_ophys': [3, 4],
            'session_type_behavior': ['foo', 'bar'],
            'session_type_ophys': ['foo', 'baz'],
            'foo': [1, 2]
        })

        with patch.object(OphysMixin, '__init__') as ophys_mixin:
            ophys_mixin.return_value = None
            mixin = OphysMixin()
        mixin._df = df
        mixin._merge_columns()

        expected = pd.DataFrame({
            'date_of_acquisition': [3, 4],
            'session_type': ['foo', 'baz'],
            'foo': [1, 2]
        })
        pd.testing.assert_frame_equal(
            expected.sort_index(axis=1),
            mixin._df.sort_index(axis=1)
        )

    def test__merge_column_values_missing(self):
        """ophys value chosen and merged with non-null behavior"""
        df = pd.DataFrame({
            'date_of_acquisition_behavior': ['foo', 'bar'],
            'date_of_acquisition_ophys': ['baz', np.nan],
        })

        with patch.object(OphysMixin, '__init__') as ophys_mixin:
            ophys_mixin.return_value = None
            mixin = OphysMixin()
        mixin._df = df
        mixin._merge_columns()

        expected = pd.DataFrame({
            'date_of_acquisition': ['baz', 'bar'],
        })
        pd.testing.assert_frame_equal(
            expected.sort_index(axis=1),
            mixin._df.sort_index(axis=1)
        )
