import scipy.signal
import numpy as np
import pandas as pd


def calc_deriv(x, time):
    dx = np.diff(x)
    dt = np.diff(time)
    dxdt_rt = np.hstack((np.nan, dx / dt))
    dxdt_lt = np.hstack((dx / dt, np.nan))
    dxdt = np.vstack((dxdt_rt, dxdt_lt))
    dxdt = np.nanmean(dxdt, axis=0)
    return dxdt


def deg_to_dist(speed_deg_per_s):
    '''
    takes speed in degrees per second
    converts to radians
    multiplies by radius (in cm) to get linear speed in cm/s
    '''
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (
        2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
    running_speed_cm_per_sec = np.pi * speed_deg_per_s * running_radius / 180.
    return running_speed_cm_per_sec


def get_running_df(data, time):
    dx_raw = data["items"]["behavior"]["encoders"][0]["dx"]
    v_sig = data["items"]["behavior"]["encoders"][0]["vsig"]
    v_in = data["items"]["behavior"]["encoders"][0]["vin"]
    assert len(v_in) == len(time), "length of v_in ({}) must match length of time ({}), they are off by {}".format(
        len(v_in), 
        len(time), 
        abs(len(v_in) - len(time))
    )
    # remove big, single frame spikes in encoder values
    dx = scipy.signal.medfilt(dx_raw, kernel_size=5)
    dx = np.cumsum(dx)  # wheel rotations
    speed = calc_deriv(dx, time)  # speed in degrees/s
    speed = deg_to_dist(speed)
    return pd.DataFrame({
        'speed': speed,
        'dx': dx_raw,
        'v_sig': v_sig,
        'v_in': v_in,
    }, index=pd.Index(time, name='timestamps'))
