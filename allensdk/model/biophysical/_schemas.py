import argparse

# Create the parser
sim_parser = argparse.ArgumentParser(description='arguments to be passed for biophysical simualtions')
sim_parser.add_argument('manifest_file',
                        help='.json configurations for running the simulations')
sim_parser.add_argument('--axon_type', required= False,
                        help='axon replacement for biophysical models')