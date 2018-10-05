import argparse

import pynwb


def main(nwb_file_path):


    with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io_obj:
        nwbfile = io_obj.read()

        units = nwbfile.units.to_dataframe()
        electrodes = nwbfile.electrodes.to_dataframe()

        print('units:')
        print(units.head())

        print('electrodes')
        print(electrodes.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='users will probably want to be able to browse tabular data')
    parser.add_argument('nwb_file_path', type=str)

    args = parser.parse_args()
    main(args.nwb_file_path)