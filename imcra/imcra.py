import numpy as np
import scipy


def mixmin(a,b):
    return min(np.amin(a),np.amin(b))

def mixmax(a,b):
    return max(np.amax(a),np.amax(b))

def find_nonzero(input):
    output = []
    for i in range(0,len(input)):
        if input[i] != 0:
            output.append(i)
    return output

def lnshift(x,t):
    length = len(x)
    if length > 1:
        y = [x[t:length], x[0:t]]
    else:
        length = len(x[0])
        y = np.concatenate((x[t:length],x[0:t]))
    return y


def imcra(input,fs):
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
    '''find a normalization factor for the window'''
    win2 = np.power(win,2)
    W0 = win2[0:frame_move]

    for k in range(0,frame_length,frame_move):
        # print(k)
        swin2 = lnshift(win2,k)
        # print(len(swin2[0:frame_move][0]))
        W0 = W0 + swin2[0][0:frame_move]

    W0 = np.mean(W0) ** 0.5
    win = win / W0
    Cwin = sum(np.power(win,2)) ** 0.5
    win = win / Cwin

    f_win_length = 1
    win_freq = np.hanning(2*f_win_length+1)  
    '''window for frequency smoothing'''
    win_freq = win_freq / sum(win_freq)  
    '''normalize the window function'''

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

    while(loop_i < data_length-frame_move):

        if(loop_i == 0):
            frame_in = input[0:frame_length]
        else:
            frame_in = np.concatenate((frame_in[frame_move:], input[loop_i:loop_i+frame_move]))

        frame_out = np.concatenate((frame_out[frame_move:], np.zeros((frame_move,1))))

        Y = np.fft.fft(frame_in*win);  

        Ya2 = np.power(abs(Y[0:N_eff]), 2)  
        '''spec estimation using single frame info.'''
        Sf = np.convolve(win_freq, Ya2)  
        '''frequency smoothing '''
        Sf = Sf[f_win_length:N_eff+f_win_length]  

        if (loop_i==0):         
            '''initialization'''
            lambda_dav = Ya2  
            ''' expected noise spec'''
            lambda_d = Ya2  
            ''' modified expected noise spec'''
            gamma = 1  
            ''' instant SNR estimation'''
            Smin = Sf  
            '''noise estimation spec value'''
            S = Sf  
            '''spec after time smoothing'''
            St = Sf  
            '''Sft:smoothing results using speech abscent probability'''
            GH1 = 1  
            '''spec gain'''
            Smint = Sf  
            '''min value get from St'''
            Smin_sw = Sf  
            '''auxiliary variable for finding min'''
            Smint_sw = Sf
            eta_2term = np.power(GH1,2) * gamma


        gamma = Ya2/max(np.amax(lambda_d), 1e-10)
        '''update instant SNR'''


        eta = alpha_eta * eta_2term + (1-alpha_eta) * max(np.amax(gamma-1), 0)
        ''' update smoothed SNR, eq.18, where eta_2term = GH1 .^ 2 .* gamma''' 
        eta = max(eta, eta_min)
        v = np.divide(gamma * eta, (1+eta))

        GH1 = np.divide(eta,(1+eta)* np.exp(0.5* scipy.special.expi(v)))
        
        S = alpha_s * S + (1-alpha_s) * Sf

        if(loop_i<(frame_length+14*frame_move)):
            Smin = S
            Smin_sw = S

        else:
            Smin = mixmin(Smin,S)
            Smin_sw = mixmin(Smin_sw, S)

        
        

        gama_min = np.divide((Ya2 / Bmin),Smin)
        zeta = np.divide(S / Bmin,Smin)

        I_f = [0]*N_eff
        for i in range(0,N_eff):
            if(gama_min[i] <gama0 and zeta[i] < zeta0):
                I_f[i] = 1
            else:
                I_f[i] = 0
            
        conv_I = np.convolve(win_freq, I_f); 
        '''smooth'''
        conv_I = conv_I[f_win_length+1:N_eff+f_win_length]
            
        Sft = St
        
        idx = find_nonzero(conv_I);   
        '''eq. 26'''
        if idx != []:
            for i in idx:
                conv_Y = np.convolve(win_freq, I_f*Ya2); 
                '''eq. 26'''
                conv_Y = conv_Y[f_win_length+1:N_eff+f_win_length]
                Sft[i] = conv_Y[i] * conv_I[i]


        St=alpha_s*St+(1-alpha_s)*Sft
        '''updated smoothed spec eq. 27'''
        
        if(loop_i<(frame_length+14*frame_move)):
            Smint = St
            Smint_sw = St
        else:
            Smint = mixmin(Smint, St)
            Smint_sw = mixmin(Smint_sw, St)
        

        
        gamma_mint = np.divide(Ya2 / Bmin, Smint)
        zetat = np.divide(S/Bmin, Smint)
        qhat = np.ones((N_eff, 1))
        '''eq. 29 speech absence probability'''
        phat = np.zeros((N_eff, 1))  
        '''eq. 29 init p(speech active|gama)'''
        
        temp = [0]*N_eff
        for i in range(0,N_eff):
            if (gamma_mint[i]>1 and gamma_mint[i]<gama1 and zetat[i]<zeta0):
                temp[i] = 1
            else:
                temp[i] = 0



        idx = find_nonzero(temp);  
        '''eq. 29'''
        for i in range(0,N_eff):
            qhat[i] = (gama1-gamma_mint[i]) / (gama1-1)
            if gamma_mint[i] >= gama1 or zetat[i] >= zeta0:
                qhat[i] = 0


         
        '''eq. 7'''
        for i in range(0,N_eff):
            phat[i] = 1/(1+qhat[i]/(1-qhat[i])*(1+eta) * np.exp(-v[i])); 
            if (gamma_mint[i] >=gama1 or zetat[i] >=zeta0):
                phat[i] = 1

        alpha_dt = alpha_d + (1-alpha_d) * phat
        lambda_dav = alpha_dt * lambda_dav + (1-alpha_dt) * Ya2
        lambda_d = lambda_dav * beta
        
        

        # loop_i = loop_i + frame_move
        # TODO: not fixed
        if l_mod_lswitch==Vwin:
            '''reinitiate every Vwin frames'''
            l_mod_lswitch=0
            if loop_i == Vwin * frame_move + 1 + frame_overlap:
                SW=repmat(S,1,Nwin)
                SWt=repmat(St,1,Nwin)
            else:
                SW=np.concatenate([SW[:,2:Nwin],Smin_sw])       
                Smin=min(SW,[],2);     
                Smin_sw=S;    
                SWt=[SWt(:,2:Nwin) Smint_sw]
                Smint=min(SWt,[],2)
                Smint_sw=St;   


    #     l_mod_lswitch = l_mod_lswitch + 1;
        
    #     gamma = Ya2 ./ max(lambda_d, 1e-10);  % update instant SNR
    #     eta = alpha_eta * eta_2term + (1-alpha_eta) * max(gamma-1, 0);  % update smoothed SNR, eq. 32 where eta_2term = GH1 .^ 2 .* gamma 
    #     eta = max(eta, eta_min);
    #     v = gamma .* eta ./ (1+eta);  
    #     GH1 = eta ./ (1+eta).*exp(0.5*expint(v));
        
    #     G = GH1 .^ phat .* GH0 .^ (1-phat);
    #     eta_2term = GH1 .^ 2 .* gamma;  % eq. 18
        
    #     X = [zeros(3,1); G(4:N_eff-1) .* Y(4:N_eff-1); 0];
    #     X(N_eff+1:frame_length) = conj(X(N_eff-1:-1:2));  % extend the anti-symmetric range of the spectum
    #     frame_result = Cwin^2*win.*real(ifft(X));
        
    #     frame_out = frame_out + frame_result;
    #     if(loop_i==1)
    #         y_out_time(loop_i:loop_i+frame_move-1) = frame_out(1:frame_move);
    #         loop_i = loop_i + frame_length;
    #     else
    #         y_out_time(loop_i-frame_overlap:loop_i+frame_move-1-frame_overlap) = frame_out(1:frame_move);
    #         loop_i = loop_i + frame_move;
    #     end
    # end
    # audiowrite('example_out.wav', y_out_time, fs);
    # audiowrite('example_in.wav', y_in_time, fs);
    








    # windowOverlap = [];
    # freqRange = 1:20000;

    # subplot(2,1,1)
    # spectrogram(y_in_time, 128, windowOverlap, freqRange, fs, 'yaxis');
    # % plot(y_in_time)
    # subplot(2,1,2)
    # spectrogram(y_out_time, 128, windowOverlap, freqRange, fs, 'yaxis');
    # % plot(y_out_time)
