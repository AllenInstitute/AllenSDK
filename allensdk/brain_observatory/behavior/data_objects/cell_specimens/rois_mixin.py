import numpy as np
import pandas as pd


class RoisMixin:
    """A mixin for a collection of rois stored as a dataframe
    (._value is a dataframe)"""
    _value: pd.DataFrame

    def filter_to_roi_ids(self, roi_ids: np.ndarray):
        """Limit to only rois given by roi_ids
        Use for, ie excluding invalid rois

        Parameters
        ----------
        roi_ids
            Filter/reorder _value to these roi_ids

        Notes
        ----------
        Will both filter and reorder dataframe to have same order as the
        input roi_ids
        """
        original_index_name = self._value.index.name
        if original_index_name is None:
            original_index_name = 'index'

        if original_index_name != 'cell_roi_id':
            self._value = (self._value
                           .reset_index()
                           .set_index('cell_roi_id'))

        if not np.in1d(roi_ids, self._value.index).all():
            raise RuntimeError(f'Not all roi ids to be filtered are in '
                               f'{type(self).__name__}')

        # Filter, reorder _value to roi_ids
        self._value = self._value.loc[roi_ids]

        if original_index_name != 'cell_roi_id':
            self._value = (self._value
                           .reset_index()
                           .set_index(original_index_name))
