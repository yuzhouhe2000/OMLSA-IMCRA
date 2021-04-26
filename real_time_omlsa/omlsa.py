# Python implementation of OMLSA 
# By Yuzhou He
# Reference: https://github.com/zhr1201/OMLSA-speech-enhancement/blob/master/myomlsa1_0.m
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time
from utils import *

############### Initialize the data ################
# data_length = len(frame_buffer)
f_win_length = 1
win_freq = np.array([0.25, 0.5, 0.25])
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
Vwin = 15
Nwin = 8
loop_i = 0
lambda_d = 0
eta_2term = 0
S = 0
St = 0
lambda_dav = 0
Smin = 0
Smint_sw = 0
Smint = 0
Smin_sw = 0
SW = 0
G = 0
conv_Y = 0
# Assume frame length = 256, frame_move = 128. Can change here
frame_length = 256
frame_move =128
N_eff = int(frame_length / 2 + 1)
frame_overlap = frame_length - frame_move
win = np.hamming(frame_length)
# win = win / (np.mean(np.power(win,2)) ** 0.5)
Cwin = sum(np.power(win,2)) ** 0.5
win = win / Cwin
N_eff = int(frame_length / 2 + 1)
data_length = frame_length
partition = frame_length/frame_move
frame_buffer = np.zeros((0, ))
frame_in = np.zeros((frame_length, ))
frame_out = np.zeros((frame_length, ))

frame_result = np.zeros((frame_length, ))
y_out_time = np.zeros((data_length, ))
l_mod_lswitch = 0
zi = np.zeros((0, ))

def omlsa_streamer(frame,fs,frame_length,frame_move,plot = None,postprocess = None,high_cut = 6000):
    global loop_i,frame_buffer,frame_out,frame_in,frame_result,y_out_time,l_mod_lswitch,lambda_d,eta_2term,S,St,lambda_dav,Smin,Smin_sw,Smint_sw,Smint,zi,G,conv_Y
    start = time.time()
    input = frame
    input = input.reshape(frame_move,)
    
    # #################### Core Algorithm ####################
    # '''OMLSA LOOP'''
    # '''For all time frames'''
    if loop_i < 1:
        loop_i  = loop_i + 1
        frame_buffer = np.concatenate((frame_buffer,input))
        return frame

    else:
        if loop_i == 1:
            frame_buffer = frame_buffer[0:128]
        else:
            frame_buffer = frame_buffer[-128:]
        # print(frame_buffer)
        frame_buffer = np.concatenate((frame_buffer,input))
        frame_in = frame_buffer
        frame_out = np.concatenate((frame_out[frame_move:], np.zeros((frame_move,))))
        Y = np.fft.fft(frame_in * win)
        Ya2 = np.power(abs(Y[0:N_eff]), 2)  
        Sf = np.convolve(win_freq.flatten(), Ya2.flatten()) 
        '''frequency smoothing '''

        Sf = Sf[f_win_length:N_eff+f_win_length]  
        
        '''initialization'''
        if (loop_i==1):         
            lambda_dav = lambda_d = Ya2  
            gamma = 1  
            GH1 = 1  
            eta_2term = np.power(GH1,2) * gamma
            S = Smin = St = Smint = Smin_sw = Smint_sw = Sf  
            
        # if (loop_i < 30) or (loop_i % 2 == 1):
        if True:
            '''instant SNR'''  
            gamma = np.divide(Ya2 ,np.maximum(lambda_d, 1e-10))
            
            ''' update smoothed SNR, eq.18, where eta_2term = GH1 .^ 2 .* gamma''' 
            eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum((gamma-1), 0)

            eta = np.maximum(eta, eta_min)
            v = np.divide(gamma * eta, (1+eta))

            GH1 = np.divide(eta,(1+eta))* np.exp(0.5* expint(v))
            
            S = alpha_s * S + (1-alpha_s) * Sf
            
            if(loop_i < 30):
                Smin = S
                Smin_sw = S

            else:
                Smin = np.minimum(Smin,S)
                Smin_sw = np.minimum(Smin_sw, S)

            gamma_min = np.divide((Ya2 / Bmin),Smin)
            zeta = np.divide(S/Bmin,Smin)

            I_f = np.zeros((N_eff, )) 
            I_f[gamma_min < gamma0] = 1
            I_f[zeta < zeta0] = 1
            
            conv_I = np.convolve(win_freq, I_f)
            
            '''smooth'''
            conv_I = conv_I[f_win_length:N_eff+f_win_length]
            
            Sft = St

            conv_Y = np.convolve(win_freq.flatten(), (I_f*Ya2).flatten())
            '''eq. 26'''
            conv_Y = conv_Y[f_win_length:N_eff+f_win_length]
            
            Sft = St
            Sft = np.divide(conv_Y,conv_I)
            Sft[(conv_I) == 0] = St[(conv_I) == 0]
            
            St=alpha_s*St+(1-alpha_s)*Sft
            '''updated smoothed spec eq. 27'''

            if(loop_i < 30):
                Smint = St
                Smint_sw = St
            else:
                Smint = np.minimum(Smint, St)
                Smint_sw = np.minimum(Smint_sw, St)
            
            gamma_mint = np.divide(Ya2/Bmin, Smint)
            zetat = np.divide(S/Bmin, Smint)
            
            '''eq. 29 speech absence probability'''
            
            '''eq. 29 init p(speech active|gama)'''
            
            temp = [0]*N_eff
                
            # find prior probability of speech presence
            qhat = (gamma1-gamma_mint) / (gamma1-1)
            qhat[gamma_mint<1] = 1
            qhat[gamma_mint<gamma1] = 1
            qhat[zetat<zeta0] = 1          
            qhat[gamma_mint >= gamma1] = 0
            qhat[zetat >= zeta0] = 0
            
            phat = np.divide(1,(1+np.divide(qhat,(1-qhat))*(1+eta) * np.exp(-v)))
            phat[gamma_mint >=gamma1] = 1
            phat[zetat >=zeta0] = 1

            alpha_dt = alpha_d + (1-alpha_d) * phat
            lambda_dav = alpha_dt * lambda_dav + (1-alpha_dt) * Ya2
            lambda_d = lambda_dav * beta
            
            if l_mod_lswitch  == 2*Vwin:
                '''reinitiate every Vwin frames'''
                l_mod_lswitch=0
                try:
                    SW=np.concatenate((SW[1:Nwin],Smin_sw))  
                    Smin=np.amin(SW); 
                    Smin_sw=S;    
                    SWt=np.concatenate((SWt[1:Nwin],Smint_sw))
                    Smint=np.amin(SWt);  
                    Smint_sw=St;  

                # initialize
                except:
                    SW= np.tile(S,(Nwin))
                    SWt= np.tile(St,(Nwin))

            l_mod_lswitch = l_mod_lswitch + 1
            
            gamma = np.divide(Ya2 , np.maximum(lambda_d, 1e-10)) 
            '''update instant SNR'''
            

            eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum(gamma-1, 0)
            
            eta[eta<eta_min] = eta_min

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

        frame_result = win * temp * Cwin * Cwin

        frame_out = frame_out + frame_result
        output,zi = bandpass(frame_out[0:frame_move],postprocess,high_cut,fs,zi)  # bandpass the signal
        # print(len(frame_out))
        loop_i = loop_i + 1

    # print(time.time()-start)
    return (output)