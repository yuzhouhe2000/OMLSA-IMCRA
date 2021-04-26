# Some helper functions
import numpy as np
import scipy.special
from scipy.signal import butter, lfilter


# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

# Circular shift of an array
def circular_shift(x,t):
    return [x[t:len(x)], x[0:t]]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass(raw_input,preprocess,high_cut,fs):
    if (preprocess == "butter"):
        input = butter_bandpass_filter(raw_input,20,high_cut,fs)
        input = input / max(abs(input))
    elif (preprocess == "ellip"):
        sos = scipy.signal.ellip(4,5,40,[20/(fs/2),high_cut/(fs/2)],btype='bandpass', output='sos')
        input = scipy.signal.sosfilt(sos, raw_input)
        input = input / max(abs(input))
    else:
        input = raw_input
    return input