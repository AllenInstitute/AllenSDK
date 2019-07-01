import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def main(csd_path, csd_timestamps_path, sigma=4, order=1):

    csd = np.load(csd_path, allow_pickle=False)

    filtered = gaussian_filter(csd, sigma, order)

    csd_timestamps = np.load(csd_timestamps_path, allow_pickle=False)
    indices = np.arange(filtered.shape[0])

    fig, ax = plt.subplots()
    plt.pcolor(csd_timestamps, indices, filtered, cmap='gray')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('csd_path', type=str)
    parser.add_argument('csd_timestamps_path', type=str)
    # parser.add_argument('csd_output_json_path', type=str)
    # parser.add_argument('probe_name', type=str)

    args = parser.parse_args()
    main(args)