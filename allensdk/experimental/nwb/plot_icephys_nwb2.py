import sys
import matplotlib.pyplot as plt
import numpy as np
import ic_ephys

DEFAULT_NWB1_FILE_NAME ="/local1/ephys/patchseq/nwb2/Npr3-IRES2-CreSst-IRES-FlpOAi65-401243.04.01.01_ver2.nwb"


def plot_sweeps(nwbfile, sweep_table,
                xlim=[0,5],
                response_offset=-100,
                stimulus_offset=-500):

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))

    trace_ix = 0
    for sweep_name, sweep_info in sweep_table.iterrows():

        acquisition = nwbfile.get_acquisition(sweep_name)
        stimulus = nwbfile.get_stimulus(sweep_name)

        acquisition_data = acquisition.data
        stimulus_data = stimulus.data

        t_acq = np.arange(0,len(acquisition_data))/acquisition.rate
        t_stm = np.arange(0,len(stimulus_data))/stimulus.rate

        stimulus_offset = -np.max(np.abs(stimulus_data))
        acquisition_offset = -np.max(np.abs(acquisition_data))

        acquisition_data += trace_ix * acquisition_offset
        stimulus_data += trace_ix * stimulus_offset

        ax[0].plot(t_stm, stimulus_data)
        ax[0].set_xlabel('time (s)')
        ax[0].text(0, trace_ix * stimulus_offset, " %s: " % sweep_name, fontsize=10)
        ax[0].set_title("Stimulus (%4.0e*%s)" % (stimulus.conversion, stimulus.unit))
        ax[0].set_xlim(t_stm[0],t_stm[-1])

        ax[1].plot(t_acq,acquisition_data)
        ax[1].set_xlabel('time (s)')
        ax[1].set_title("Acquisition (%4.0e*%s)" % (acquisition.conversion,acquisition.unit))
        ax[1].set_xlim(t_acq[0],t_acq[-1])
        ax[1].text(0, trace_ix * acquisition_offset, " %s: " % sweep_name, fontsize=10)

        trace_ix+=1

    fig.suptitle(sweep_info["stimulus_description"], fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])





def main():

    """
    # Usage:
    $ python plot_icephys_nwb2.py NWB2_FILE_NAME
    example files:"/local1/ephys/patchseq/nwb2/Pvalb-IRES-Cre;Ai14-406663.04.01.01_ver2.nwb"

    """
    if len(sys.argv) == 1:
        sys.argv.append(DEFAULT_NWB2_FILE_NAME)

    nwb2_file_name = sys.argv[1]

    nwbfile = ic_ephys.load_nwb2_file(nwb2_file_name)

    sweep_table = ic_ephys.build_sweep_table(nwbfile)

    for name, sweep_group_table in sweep_table.groupby("stimulus_description"):
        plot_sweeps(nwbfile,sweep_group_table)

    plt.show()


if __name__ == "__main__": main()
