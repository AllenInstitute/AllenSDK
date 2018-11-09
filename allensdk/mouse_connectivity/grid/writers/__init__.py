from .write_classic_grid_files import write_classic_grid_files

def get_writer(case):
    if case == 'classic':
        return write_classic_grid_files
    else:
        raise ValueError('unrecognized case: {}'.format(case))