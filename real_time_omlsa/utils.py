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


# Faster for loop usinig numba
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
    for i in range(0,N_eff):
        if (gamma_mint[i]>1 and gamma_mint[i]<gamma1 and zetat[i]<zeta0):
            qhat[i] = (gamma1-gamma_mint[i]) / (gamma1-1)
        elif gamma_mint[i] >= gamma1 or zetat[i] >= zeta0:
            qhat[i] = 0
    return qhat

@njit(fastmath=True, cache=True)
def update_eta(N_eff,eta,eta_min,alpha_eta,eta_2term,gamma):
    eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum(gamma-1, 0)
    '''update smoothed SNR, eq. 32 where eta_2term = GH1 .^ 2 .* gamma '''
    for i in range(0,N_eff):
        if eta[i] < eta_min:
            eta[i] = eta_min
    return eta

@njit(fastmath=True, cache=True)
def find_phat(N_eff,gamma_mint,gamma1,zetat,zeta0,v,eta,qhat):
    phat = np.zeros((N_eff, ))  
    phat = np.divide(1,(1+np.divide(qhat,(1-qhat))*(1+eta) * np.exp(-v)))
    for i in range(0,N_eff):  
        if (gamma_mint[i] >=gamma1 or zetat[i] >=zeta0):
            phat[i] = 1
    return phat


@njit(fastmath=True, cache=True)
def find_I_f(N_eff,gamma0,zeta,zeta0,gamma_min):
    I_f = [0]*N_eff
    for i in range(0,N_eff):
        if(gamma_min[i] <gamma0 and zeta[i] < zeta0):
            I_f[i] = 1
        else:
            I_f[i] = 0
    return I_f


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
