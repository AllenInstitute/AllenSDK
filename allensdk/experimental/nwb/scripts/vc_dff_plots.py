from pynwb import NWBHDF5IO
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def plot_dff_vs_speed(dff, dff_t, speed, speed_t):
    fig, ax = plt.subplots(figsize=(8, 4))
    axes = [ax, ax.twinx(), ax.twinx()]
    fig.subplots_adjust(right=0.9)
    axes[-1].spines['right'].set_position(('axes', 1.2))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    axes[0].plot(dff_t, dff, "b")
    axes[0].set_ylabel("dF/F", color="b")
    axes[0].set_xlabel("time (s)")
    axes[1].plot(speed_t, speed, "r")
    axes[1].set_ylabel("speed (cm/s)", color="r")
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("nwb_file", type=str)
    args = parser.parse_args()

    with NWBHDF5IO(args.nwb_file, mode='r') as io:
        nwb = io.read()
        running_series = nwb.get_acquisition('running_speed')
        processing = nwb.get_processing_module('visual_coding_pipeline')
        interface = processing.get_data_interface('dff_interface')
        dff_series = interface.roi_response_series['df_over_f']
        plot_dff_vs_speed(dff_series.data[4,:], dff_series.timestamps,
                          running_series.data, running_series.timestamps)


if __name__ == "__main__":
    main()
