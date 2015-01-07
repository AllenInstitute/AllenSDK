from allen_wrench.core.orca_data_set import OrcaDataSet as EphysDataSet

def load_sweeps(file_name, sweep_numbers):
    data = [ load_sweep(file_name, sweep_number) for sweep_number in sweep_numbers ]

    return {
        'voltage': [ d['voltage'] for d in data ],
        'current': [ d['current'] for d in data ],
        'dt': [ d['dt'] for d in data ],
        'start_idx': [ d['start_idx'] for d in data ],
    }
    
def load_sweep(file_name, sweep_number):
    ds = EphysDataSet(file_name)
    data = ds.get_full_sweep(sweep_number)

    return {
        'current': data['stimulus'],
        'voltage': data['response'],
        'start_idx': data['index_range'][0],
        'dt': 1.0/data['sampling_rate']
    }
