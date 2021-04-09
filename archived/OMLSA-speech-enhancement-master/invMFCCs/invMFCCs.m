clc;clear all; 
infilename = 'INLS1.wav';
[speech.clean,fs] = wavread(infilename);
wav = speech.clean;
L = 256; % frame length
nfft = 256;% DFT size
hopfactor = 2;
inc = L-L/hopfactor; % frame advance
hopfactor = L/(L-inc);
InterpMultiple = 1;

stopdB = 150;
a       = 0.50;
b       = -0.50;
n       = 1:L;
S       = L/hopfactor; %hop size
MaxIter = 300;

win     = sqrt(S)/sqrt((4*a^2+2*b^2)*L)*(a+b*cos(2*pi*n/L));
win     = win(:);
window  = win;
Y=stft(wav,window,inc,nfft);
Y=abs(Y);
%% Specify the number of Mel filter bands
MelBankVec = [10 20 30 40 50 60 70];

for MelBankIndex =1:length(MelBankVec)
    MelBankNum = MelBankVec(MelBankIndex);
    [Y_rec,mfcc] = MelCompress(Y,MelBankNum,fs);    
    %% Interp the reconstructed amplitude spectrum
    Y_Interp=InterpMfcc(Y_rec,InterpMultiple);
    hopfactor = hopfactor*InterpMultiple;
    Y_rec = Y_Interp;
    %% Reconstruct the speech signal from its spectral spectrum via the LSE-ISTFTM algorithm
    [rec , D] = LSEE(Y_rec,window,hopfactor,nfft,stopdB,MaxIter);
    outfilename = strcat('INLS1_IRLS_mel_len256_inc128_melbank_',num2str(MelBankNum),'_iter',num2str(MaxIter),'.wav');
    wavwrite(rec/max(abs(rec)),fs,outfilename);
end

