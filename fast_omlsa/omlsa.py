# Python implementation of OMLSA 
# By Yuzhou He
# Reference: https://github.com/zhr1201/OMLSA-speech-enhancement/blob/master/myomlsa1_0.m
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import scipy.special
from numba import njit, prange
from scipy.signal import butter, lfilter
# Some helper functions

# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

# Circular shift of an array
def circular_shift(x,t):
    return [x[t:len(x)], x[0:t]]


# Faster for loop usinig numba
@njit(fastmath=True, cache=True)
def fast_loop1(N_eff,conv_Y,conv_I,Sft):
    for i in range(0,N_eff):
        if int(conv_I[i]) != 0:
            Sft[i] = np.divide(conv_Y[i],conv_I[i])
    return Sft

@njit(fastmath=True, cache=True)
def fast_loop2(N_eff,gamma_mint,gamma1,zeta0,zetat,qhat):
    for i in range(0,N_eff):
        if (gamma_mint[i]>1 and gamma_mint[i]<gamma1 and zetat[i]<zeta0):
            qhat[i] = (gamma1-gamma_mint[i]) / (gamma1-1)
        elif gamma_mint[i] >= gamma1 or zetat[i] >= zeta0:
            qhat[i] = 0
    return qhat

@njit(fastmath=True, cache=True)
def fast_loop3(N_eff,eta,eta_min):
    for i in range(0,N_eff):
        if eta[i] < eta_min:
            eta[i] = eta_min
    return eta

@njit(fastmath=True, cache=True)
def fast_loop4(N_eff,gamma_mint,gamma1,zetat,zeta0,phat):
    for i in range(0,N_eff):  
        if (gamma_mint[i] >=gamma1 or zetat[i] >=zeta0):
            phat[i] = 1
    return phat


njit(fastmath=True, cache=True)
def fast_loop5(N_eff,I_f,gamma0,zeta,zeta0,gamma_min):
    for i in range(0,N_eff):
        if(gamma_min[i] <gamma0 and zeta[i] < zeta0):
            I_f[i] = 1
        else:
            I_f[i] = 0
    return I_f


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


# OMLSA + IMCRA algorithm
def omlsa(raw_input,fs,frame_length,frame_move,plot = None,preprocess = None,high_cut = None):
    start = time.time()

    if (preprocess == "butter"):

        input = butter_bandpass_filter(raw_input,20,high_cut,fs)
        input = input / max(abs(input))

    elif (preprocess == "ellip"):
        sos = scipy.signal.ellip(4,5,40,[20/(fs/2),high_cut/(fs/2)],btype='bandpass', output='sos')
        input = scipy.signal.sosfilt(sos, raw_input)
        input = input / max(abs(input))
    else:
        input = raw_input
    data_length = len(input)
    frame_overlap = frame_length - frame_move
    N_eff = int(frame_length / 2 + 1)
    loop_i = 0
    frame_in = np.zeros((frame_length, ))
    frame_out = np.zeros((frame_length, ))
    frame_result = np.zeros((frame_length, ))
    y_out_time = np.zeros((data_length, ))
    win = np.hamming(frame_length)
    '''normalization of window'''
    win2 = np.power(win,2)
    W0 = win2[0:frame_move]

    for t in range(0,frame_length,frame_move):
        '''circular shift for weight calculation'''
        swin2 = circular_shift(win2,t)
        W0 = W0 + swin2[0][0:frame_move]

    W0 = np.mean(W0) ** 0.5
    win = win / W0

    Cwin = sum(np.power(win,2)) ** 0.5
    win = win / Cwin

    f_win_length = 1 
    
    win_freq = [0.25, 0.5, 0.25]
    win_freq = np.array(win_freq)
    '''window for frequency smoothing'''
    '''normalize the window function'''

    '''smoothing parameter'''
    alpha_eta = 0.92
    alpha_s = 0.9
    alpha_d = 0.85
    
    beta = 2
    eta_min = 0.0158
    GH0 = np.power(eta_min,0.5)
    gamma0 = 4.6
    gamma1 = 3
    zeta0 = 1.67
    Bmin = 1.66
    l_mod_lswitch = 0
    Vwin = 15
    Nwin = 8

    '''OMLSA LOOP'''
    '''For all time frames'''
    while(loop_i+frame_length < data_length):
        '''if is first iteration, initialize all the variables'''
        if(loop_i == 0):
            frame_in = input[0:frame_length]
        else:
            '''move the frame by step = frame_moove'''
            frame_in = np.concatenate((frame_in[frame_move:], input[loop_i:loop_i+frame_move]))

        frame_out = np.concatenate((frame_out[frame_move:], np.zeros((frame_move,))))

        Y = np.fft.fft(frame_in*win)

        Ya2 = np.power(abs(Y[0:N_eff]), 2)  

        '''spec estimation using single frame info.'''
        Sf = np.convolve(win_freq.flatten(), Ya2.flatten())  

        '''frequency smoothing '''
        Sf = Sf[f_win_length:N_eff+f_win_length]  

        '''initialization'''
        if (loop_i==0):         
            
            ''' 
            lambda_dav = expected noise spec
            lambda_d = modified expected noise spec
            '''
            
            lambda_dav = lambda_d = Ya2  
            
            ''' instant SNR estimation'''
            gamma = 1  
            
            '''
            S = spec after time smoothing
            Smin = noise estimation spec value
            Sft = smoothing results using speech abscent probability
            Smint = min value get from St
            Smin_sw = auxiliary variable for finding min
            '''
            S = Smin = St = Smint = Smin_sw = Smint_sw = Sf  

            '''spec gain'''
            GH1 = 1  
            
            eta_2term = np.power(GH1,2) * gamma

        

        '''instant SNR'''  
        gamma = np.divide(Ya2 ,np.maximum(lambda_d, 1e-10))
        
        
        ''' update smoothed SNR, eq.18, where eta_2term = GH1 .^ 2 .* gamma''' 
        eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum((gamma-1), 0)

        eta = np.maximum(eta, eta_min)
        v = np.divide(gamma * eta, (1+eta))

        GH1 = np.divide(eta,(1+eta))* np.exp(0.5* expint(v))
        
        S = alpha_s * S + (1-alpha_s) * Sf
        
        if(loop_i<(frame_length+14*frame_move)):
            Smin = S
            Smin_sw = S

        else:
            Smin = np.minimum(Smin,S)
            Smin_sw = np.minimum(Smin_sw, S)

        gamma_min = np.divide((Ya2 / Bmin),Smin)
        zeta = np.divide(S/Bmin,Smin)

        I_f = [0]*N_eff
        
        I_f = fast_loop5(N_eff,I_f,gamma0,zeta,zeta0,gamma_min)

        conv_I = np.convolve(win_freq, I_f)
        
        '''smooth'''
        conv_I = conv_I[f_win_length:N_eff+f_win_length]
        

        Sft = St
        
        '''eq. 26'''       
        conv_Y = np.convolve(win_freq.flatten(), (I_f*Ya2).flatten())

        '''eq. 26'''
        conv_Y = conv_Y[f_win_length:N_eff+f_win_length]

        Sft = fast_loop1(N_eff,conv_Y,conv_I,Sft)
             
        St=alpha_s*St+(1-alpha_s)*Sft
        '''updated smoothed spec eq. 27'''
        
        if(loop_i<(frame_length+14*frame_move)):
            Smint = St
            Smint_sw = St
        else:
            Smint = np.minimum(Smint, St)
            Smint_sw = np.minimum(Smint_sw, St)
        
        gamma_mint = np.divide(Ya2/Bmin, Smint)
        zetat = np.divide(S/Bmin, Smint)
        qhat = np.ones((N_eff, ))
        '''eq. 29 speech absence probability'''
        phat = np.zeros((N_eff, ))  
        '''eq. 29 init p(speech active|gama)'''
        
        temp = [0]*N_eff

        qhat = fast_loop2(N_eff,gamma_mint,gamma1,zeta0,zetat,qhat)
        
        phat = np.divide(1,(1+np.divide(qhat,(1-qhat))*(1+eta) * np.exp(-v)))

        phat = fast_loop4(N_eff,gamma_mint,gamma1,zetat,zeta0,phat)

        alpha_dt = alpha_d + (1-alpha_d) * phat
        lambda_dav = alpha_dt * lambda_dav + (1-alpha_dt) * Ya2
        lambda_d = lambda_dav * beta
        

        if l_mod_lswitch==Vwin:
            '''reinitiate every Vwin frames'''
            l_mod_lswitch=0
            if loop_i == Vwin * frame_move + frame_overlap:
                SW= np.tile(S,(Nwin))
                SWt= np.tile(St,(Nwin))
            else:
                SW=np.concatenate((SW[1:Nwin],Smin_sw))  
                Smin=np.amin(SW); 
                Smin_sw=S;    
                SWt=np.concatenate((SWt[1:Nwin],Smint_sw))
                Smint=np.amin(SWt);  
                Smint_sw=St;   

        l_mod_lswitch = l_mod_lswitch + 1
        
        gamma = np.divide(Ya2 , np.maximum(lambda_d, 1e-10)) 
        '''update instant SNR'''

        eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum(gamma-1, 0)
        '''update smoothed SNR, eq. 32 where eta_2term = GH1 .^ 2 .* gamma '''

        eta = fast_loop3(N_eff,eta,eta_min)

        v = np.divide(gamma * eta , (1+eta))

        GH1 = np.divide(eta , (1+eta))* np.exp(0.5 * expint(v))
        
        G = np.power(GH1 , phat) * np.power(GH0 , (1-phat))
        
        eta_2term = np.power(GH1 , 2) * gamma  
        '''eq. 18'''

        X = np.concatenate((np.zeros((3,)), (G[3:N_eff-1])*(Y[3:N_eff-1]),[0]))

        X_2 = X[1:N_eff-1]
        X_2 = X_2[::-1]
        X_other_half = np.conj(X_2) 
        X = np.concatenate((X,X_other_half))

        '''extend the anti-symmetric range of the spectum'''
        temp = np.real(np.fft.ifft(X))

        frame_result = np.power(Cwin,2) * win * temp

        frame_out = frame_out + frame_result
        
        if(loop_i==0):
            y_out_time[loop_i:loop_i+frame_move] = frame_out[0:frame_move]
            loop_i = loop_i + frame_length
        else:
            y_out_time[loop_i-frame_overlap:loop_i+frame_move-frame_overlap] = frame_out[0:frame_move]
            loop_i = loop_i + frame_move

    y_out_time = y_out_time
    print(time.time()-start)
    
    # Choose between plot strategy
    if plot == "f":
        NFFT = 256
        fig, axes = plt.subplots(nrows=2, ncols=1)
        Pxx, freqs, bins, im = axes[0].specgram(raw_input,NFFT=NFFT, Fs=fs, noverlap = NFFT/2)
        Pxx, freqs, bins, im = axes[1].specgram(y_out_time,NFFT=NFFT, Fs=fs, noverlap= NFFT/2)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    if plot == "t":
        plt.subplot(2,1,1)
        plt.plot(input)
        plt.subplot(2,1,2)
        plt.plot(y_out_time)
        plt.show()

    return (y_out_time)