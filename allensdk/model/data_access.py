from allensdk.core.nwb_data_set import NwbDataSet
from scipy import signal
import np as np

def load_sweep(file_name, sweep_number, dt=None, cut=0, bessel=False):
    ds = NwbDataSet(file_name)
    data = ds.get_sweep(sweep_number)

    data["dt"] = 1.0 / data["sampling_rate"]

    if cut > 0:
        data["response"] = data["response"][cut:]
        data["stimulus"] = data["stimulus"][cut:]        

    if bessel:
        b, a = signal.bessel(bessel["N"], bessel["Wn"], "low")
        data['response'] = signal.filtfilt(b, a, data['response'], axis=0)

    if dt is not None:
        if data["dt"] != dt:
            data["response"] = subsample_data(data["response"], "mean", data["dt"], dt)
            data["stimulus"] = subsample_data(data["stimulus"], "mean", data["dt"], dt)
            data["start_idx"] = int(data["index_range"][0] / (dt / data["dt"]))
            data["dt"] = dt

    if "start_idx" not in data:
        data["start_idx"] = data["index_range"][0]

    return {
        "voltage": data["response"],
        "current": data["stimulus"],
        "dt": data["dt"],
        "start_idx": data["start_idx"]
        }


def load_sweeps(file_name, sweep_numbers, dt=None, cut=0, bessel=False):
    data = [ load_sweep(file_name, sweep_number, dt, cut, bessel) for sweep_number in sweep_numbers ]

    return {
        'voltage': [ d['voltage'] for d in data ],
        'current': [ d['current'] for d in data ],
        'dt': [ d['dt'] for d in data ],
        'start_idx': [ d['start_idx'] for d in data ],
        } 


def subsample_data(data, method, present_time_step, desired_time_step):
    if present_time_step > desired_time_step:
        raise Exception("you desired times step is smaller than your current time step")

    # number of elements to average over
    n = int(desired_time_step / present_time_step)

    data_subsampled = None

    if method == "mean":
        # if n does not divide evenly into the length of the array, crop off the end
        end = n * int(len(data) / n)
            
        return np.mean(data[:end].reshape(-1,n), 1)

    raise Exception("unknown subsample method: %s" % (method))