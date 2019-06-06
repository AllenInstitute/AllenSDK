import warnings

import numpy as np

from . import barcode
from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import (
    EcephysSyncDataset,
)


class BarcodeSyncDataset(EcephysSyncDataset):
    @property
    def barcode_line(self):
        """ Obtain the index of the barcode line for this dataset.

        """

        if "barcode" in self.line_labels:
            return self.line_labels.index("barcode")
        elif "barcodes" in self.line_labels:
            return self.line_labels.index("barcodes")
        else:
            raise ValueError("no barcode line found")

    def extract_barcodes(self, **barcode_kwargs):
        """ Read barcodes and their times from this dataset's barcode line.

        Parameters
        ----------
        **barcode_kwargs : 
            Will be passed to .barcode.extract_barcodes_from_times

        Returns 
        -------
        times : np.ndarray
            The start times of each detected barcode.
        codes : np.ndarray
            The values of each detected barcode

        """

        sample_freq_digital = float(self.sample_frequency)
        barcode_channel = self.barcode_line

        on_events = self.get_rising_edges(barcode_channel)
        off_events = self.get_falling_edges(barcode_channel)

        on_times = on_events / sample_freq_digital
        off_times = off_events / sample_freq_digital

        return barcode.extract_barcodes_from_times(
            on_times, off_times, **barcode_kwargs
        )

    def get_barcode_table(self, **barcode_kwargs):
        """ A convenience method for getting barcode times and codes in a dictionary.

        Notes
        -----
        This method is deprecated! 

        """
        warnings.warn(
            np.VisibleDeprecationWarning(
                "This function is deprecated as unecessary (and slated for removal). Instead, simply use extract_barcodes."
            )
        )

        barcode_times, barcodes = self.extract_barcodes(**barcode_kwargs)
        return {"codes": barcodes, "times": barcode_times}
