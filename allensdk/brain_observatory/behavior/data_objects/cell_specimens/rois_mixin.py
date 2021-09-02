import warnings

import numpy as np
import pandas as pd


class RoisMixin:
    """A mixin for a collection of rois stored as a dataframe
    (._value is a dataframe)"""
    _value: pd.DataFrame

    def order_rois(self, roi_ids: np.ndarray, raise_if_rois_missing=True):
        """Orders dataframe according to input roi_ids.
        Will also filter dataframe to contain only rois given by roi_ids.
        Use for, ie excluding invalid rois

        Parameters
        ----------
        roi_ids
            Filter/reorder _value to these roi_ids
        raise_if_rois_missing
            Whether to raise exception if there are rois in the input roi_ids
            not in the dataframe

        Notes
        ----------
        Will both filter and reorder dataframe to have same order as the
        input roi_ids.

        If there are values in the input roi_ids that are not in the dataframe,
        then these roi ids will be ignored and a warning will be logged.

        Raises
        ----------
        RuntimeError if raise_if_rois_missing and there are input roi_ids not
        in dataframe
        """
        original_index_name = self._value.index.name
        if original_index_name is None:
            original_index_name = 'index'

        if original_index_name != 'cell_roi_id':
            self._value = (self._value
                           .reset_index()
                           .set_index('cell_roi_id'))

        # Reorders dataframe according to roi_ids
        self._value = self._value.reindex(roi_ids)

        is_na = self._value.isna().any(axis=0)

        if is_na.any():
            # There are some roi ids in input not in index.

            msg = f'Input contains roi ids not in ' \
                  f'{type(self).__name__}.'
            if raise_if_rois_missing:
                raise RuntimeError(msg)
            warnings.warn(msg)

            # Drop rows where NaN
            self._value = self._value.dropna(axis=0)

        if original_index_name != 'cell_roi_id':
            self._value = (self._value
                           .reset_index()
                           .set_index(original_index_name))
            if original_index_name == 'index':
                # Set it back to None
                self._value.index.name = None
