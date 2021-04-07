# Python implementation of OMLSA 
# By Yuzhou He
# Reference: https://github.com/zhr1201/OMLSA-speech-enhancement/blob/master/myomlsa1_0.m
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import scipy.special

# Some helper functions

# Turn nparray from size (x,) to (x,1)
def reformat(input):
    input = input.reshape(len(input),1)
    return input

# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

# Circular shift of an array
def circular_shift(x,t):
    return [x[t:len(x)], x[0:t]]

# OMLSA + IMCRA algorithm
def omlsa(input,fs,plot = None):
    start = time.time()
    data_length = len(input)
    frame_length = 512
    frame_move = 128
    frame_overlap = frame_length - frame_move
    N_eff = int(frame_length / 2 + 1)
    loop_i = 0
    frame_in = np.zeros((frame_length, 1))
    frame_out = np.zeros((frame_length, 1))
    frame_result = np.zeros((frame_length, 1))
    y_out_time = np.zeros((data_length, 1))
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
    gama0 = 4.6
    gama1 = 3
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

        frame_out = np.concatenate((frame_out[frame_move:], np.zeros((frame_move,1))))

        Y = np.fft.fft(frame_in*win)
        Y = reformat(Y)

        Ya2 = np.power(abs(Y[0:N_eff]), 2)  
        '''spec estimation using single frame info.'''
        Sf = Ya2
        # Sf = np.convolve(win_freq.flatten(), Ya2.flatten())  
        # Sf = reformat(Sf)
        '''frequency smoothing '''
        # Sf = Sf[f_win_length:N_eff+f_win_length]  

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

        gama_min = np.divide((Ya2 / Bmin),Smin)
        zeta = np.divide(S/Bmin,Smin)

        I_f = [0]*N_eff
        
        
        # about 0.2 second
        for i in range(0,N_eff):
            if(gama_min[i] <gama0 and zeta[i] < zeta0):
                I_f[i] = 1
            else:
                I_f[i] = 0
                
        

        # conv_I = I_f
        conv_I = np.convolve(win_freq, I_f)
        conv_I = reformat(conv_I)
        '''smooth'''
        conv_I = conv_I[f_win_length:N_eff+f_win_length]


        Sft = St
        # idx = find_nonzero(conv_I)
        I_f = reformat(np.array(I_f))
        
        '''eq. 26'''

            
        conv_Y = np.convolve(win_freq.flatten(), (I_f*Ya2).flatten())
        conv_Y = reformat(conv_Y) 
        '''eq. 26'''
        conv_Y = conv_Y[f_win_length:N_eff+f_win_length]

        # about 0.4s
        for i in range(0,N_eff):
            if conv_I[i] != 0:
                Sft[i] = np.divide(conv_Y[i],conv_I[i])
        
        
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
        qhat = np.ones((N_eff, 1))
        '''eq. 29 speech absence probability'''
        phat = np.zeros((N_eff, 1))  
        '''eq. 29 init p(speech active|gama)'''
        
        temp = [0]*N_eff

        
        # about 0.3s
        for i in range(0,N_eff):
            if (gamma_mint[i]>1 and gamma_mint[i]<gama1 and zetat[i]<zeta0):
                qhat[i] = (gama1-gamma_mint[i]) / (gama1-1)
            elif gamma_mint[i] >= gama1 or zetat[i] >= zeta0:
                qhat[i] = 0

        phat = np.divide(1,(1+np.divide(qhat,(1-qhat))*(1+eta) * np.exp(-v)))

        # about 0.2s
        for i in range(0,N_eff):  
            if (gamma_mint[i] >=gama1 or zetat[i] >=zeta0):
                phat[i] = 1
        
        alpha_dt = alpha_d + (1-alpha_d) * phat

        lambda_dav = alpha_dt * lambda_dav + (1-alpha_dt) * Ya2

        lambda_d = lambda_dav * beta


        if l_mod_lswitch==Vwin:
            '''reinitiate every Vwin frames'''
            l_mod_lswitch=0
            if loop_i == Vwin * frame_move + frame_overlap:
                SW= np.tile(S,(1,Nwin))
                SWt= np.tile(St,(1,Nwin))
            else:
                SW=np.concatenate((SW[:,1:Nwin],Smin_sw),axis = 1)  
                Smin=np.amin(SW,axis=1); 
                Smin = reformat(Smin) 
                Smin_sw=S;    
                SWt=np.concatenate((SWt[:,1:Nwin],Smint_sw),axis = 1)
                Smint=np.amin(SWt,axis=1);  
                Smint = reformat(Smint)
                Smint_sw=St;   

        l_mod_lswitch = l_mod_lswitch + 1

        gamma = np.divide(Ya2 , np.maximum(lambda_d, 1e-10)) 
        '''update instant SNR'''
        
        eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum(gamma-1, 0)
        '''update smoothed SNR, eq. 32 where eta_2term = GH1 .^ 2 .* gamma '''

        # about 0.1s
        for i in range(0,N_eff):
            if eta[i] < eta_min:
                eta[i] = eta_min
        

        v = np.divide(gamma * eta , (1+eta))

        GH1 = np.divide(eta , (1+eta))* np.exp(0.5 * expint(v))
        
        G = np.power(GH1 , phat) * np.power(GH0 , (1-phat))
        
        eta_2term = np.power(GH1 , 2) * gamma  
        '''eq. 18'''

        X = np.concatenate((np.zeros((3,1)), (G[3:N_eff-1])*(Y[3:N_eff-1]),[[0]]))

        X_2 = X[1:N_eff-1]
        X_2 = X_2[::-1]
        X_other_half = np.conj(X_2) 
        X = np.concatenate((X,X_other_half))

        '''extend the anti-symmetric range of the spectum'''
        X = X.reshape(len(X),)
        temp = np.real(np.fft.ifft(X))

        frame_result = np.power(Cwin,2) * win * temp
        frame_result = frame_result.reshape(len(frame_result),1)

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
        Pxx, freqs, bins, im = axes[0].specgram(input,NFFT=NFFT, Fs=fs, noverlap = NFFT/2, vmin= -80)
        y_out_time_reshape  = y_out_time.reshape(len(y_out_time),)
        Pxx, freqs, bins, im = axes[1].specgram(y_out_time_reshape,NFFT=NFFT, Fs=fs, noverlap= NFFT/2,vmin= -80)
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