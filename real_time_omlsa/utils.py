# Some helper functions
import numpy as np
import scipy.special
from numba import njit, prange
from scipy.signal import butter, lfilter


# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

def butter_bandpass_filter(data,lowcut, highcut, fs, zi,order=4):
    low = lowcut /(fs/2)
    high = highcut /(fs/2)
    sos = butter(order,high, btype='low',output = 'sos')
    if len(zi) == 0:
        zi = scipy.signal.sosfilt_zi(sos)
    # print(zi)
    data = data.reshape(len(data),)
    y,zi = scipy.signal.sosfilt(sos, data,zi = zi)
    return y,zi

def bandpass(raw_input,preprocess,high_cut,fs,zi):
    if (preprocess == "butter"):
        input,zi = butter_bandpass_filter(raw_input,20,high_cut,fs,zi)
        
    else:
        input = raw_input
        zi = np.zeros((0,))

    return input,zi
