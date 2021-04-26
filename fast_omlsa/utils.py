# Some helper functions
import numpy as np
import scipy.special
from numba import njit, prange
from scipy.signal import butter, lfilter


# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

# Circular shift of an array
def circular_shift(x,t):
    return [x[t:len(x)], x[0:t]]

@njit(fastmath=True, cache=True)
def find_Sft(N_eff,conv_Y,conv_I,St):
    Sft = St
    for i in range(0,N_eff):
        if int(conv_I[i]) != 0:
            Sft[i] = np.divide(conv_Y[i],conv_I[i])
    return Sft

@njit(fastmath=True, cache=True)
def find_qhat(N_eff,gamma_mint,gamma1,zeta0,zetat):
    qhat = np.ones((N_eff, ))
    # for i in range(0,N_eff):
    #     if (gamma_mint[i]>1 and gamma_mint[i]<gamma1 and zetat[i]<zeta0):
    #         qhat[i] = (gamma1-gamma_mint[i]) / (gamma1-1)
    # print(qhat)
    return qhat

@njit(fastmath=True, cache=True)
def find_phat(N_eff,gamma_mint,gamma1,zetat,zeta0,v,eta,qhat):
    phat = np.zeros((N_eff, ))  
    phat = np.divide(1,(1+np.divide(qhat,(1-qhat))*(1+eta) * np.exp(-v)))
    for i in range(0,N_eff):  
        if (gamma_mint[i] >=gamma1 or zetat[i] >=zeta0):
            phat[i] = 1
    # print(phat)
    return phat

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