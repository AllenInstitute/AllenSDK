from pynwb import NWBHDF5IO

def save_nwb2_file(nwb2file,nwb2_file_name):

    """

    Parameters
    ----------
    nwb2file: pynwb.NWBFile Class

    nwb2_file_name: string
        output file name

    Returns
    -------

    """
    io = NWBHDF5IO(nwb2_file_name, 'w')
    io.write(nwb2file)
    io.close()


def load_nwb2_file(nwb2_file_name):
    """

    Parameters
    ----------
    nwb2_file_name: string

    Returns
    -------
    nwbfile: pynwb.NWBFile Class

    """
    io = NWBHDF5IO(nwb2_file_name, 'r')

    return io.read()

