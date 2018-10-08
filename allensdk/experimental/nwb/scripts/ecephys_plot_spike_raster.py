import argparse

import numpy as np
import matplotlib.pyplot as plt
import pynwb


def main(nwb_file_path, min_time, max_time, bin_width, max_per_bin):


    with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io_obj:
        nwbfile = io_obj.read()

        units = nwbfile.units.to_dataframe()
        spike_times = nwbfile.modules['spike_detection'].data_interfaces['UnitTimes']
        unit_positions = list(spike_times.unit_ids.data)

        for jj, current_probe_units in units.groupby('probe_id'):

            if max_time is None:
                max_time = np.amax(spike_times.spike_times.data[:])
            if min_time is None:
                min_time = np.amin(spike_times.spike_times.data[:])
            bins = np.arange(min_time, max_time + bin_width, bin_width)

            fig, ax = plt.subplots()

            counts = []
            for ii, (unit_id, unit_row) in enumerate(current_probe_units.iterrows()):
                unit_index = unit_positions.index(unit_id)
                unit_times = spike_times.get_unit_spike_times(unit_index)
                unit_hist, _ = np.histogram(unit_times, bins)
                counts.append(unit_hist)
            counts = np.array(counts)
            

            plt.imshow(counts, interpolation='none', cmap=plt.cm.afmhot, vmin=0, vmax=max_per_bin)

            plt.xticks([0, counts.shape[1]-1], [str(min_time), str(max_time)])
            ax.set_xlabel('time (s)')

            ax.set_ylabel('unit id')

            probe_id = current_probe_units['probe_id'].values[0]
            ax.set_title('spike raster for probe {} on session {}'.format(probe_id, nwbfile.identifier))

            plt.axis('tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='produce images of time-binned spike counts on each probe')
    parser.add_argument('nwb_file_path', type=str)
    parser.add_argument('--min_time', type=float, default=None, help='units are seconds')
    parser.add_argument('--max_time', type=float, default=None, help='units are seconds')
    parser.add_argument('--bin_width', type=float, default=0.01, help='units are seconds')
    parser.add_argument('--max_per_bin', type=int, default=5.0, help='ceiling for spike counts in a bin')

    args = parser.parse_args()
    main(args.nwb_file_path, args.min_time, args.max_time, args.bin_width, args.max_per_bin)