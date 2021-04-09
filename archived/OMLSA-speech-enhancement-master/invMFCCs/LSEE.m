function [rec , D] = LSEE(y1,window,hopfactor,nfft,stopdB,MaxIter)    
% [rec , D] = LSEE(y1,window,hopfactor)
% Least Squares Error Reconstruction method
% y1 - Spectal Magnitudes matrix
% window - analysis window assumed for STFT
% hopfactor - number of analysis hops within one window
% (c) Shlomo Dubnov sdubnov@ucsd.edu

fs = 8000;
if nargin == 3,
    stopdB = 100;
end
D = [];
[m1 n1] = size(y1);
window_length = length(window);
step_dist = window_length/hopfactor;
overlap = window_length - step_dist;

if isreal(y1),
    %generate the Initial Estimate according to Normal Distribution :
    cols = n1*step_dist+window_length;
    curr = randn(1,cols);
else
    %resynth[esize the signal with approximate initial phase as given by y1
    curr = istftw2(y1,hopfactor,window);
    y1 = abs(y1);
end

for(pp = 1:MaxIter)
    %compute STFT :
    curr_t = curr';
    y2 = stft(curr_t,window,overlap);
    
    %let's subtitute the amplitude with given magnitude :
%     y6 = y2./abs(y2);
%     y3 = (y1 .* y6);
    theta = angle(y2);
    y3 = y1 .* exp(i*theta);
    
    old = curr;
    curr = istftw2(y3,hopfactor,window);
    curr = real(curr);
    
    err = sum((old-curr).^2);
    sig = sum(curr.^2);
    errdB = 20*log10(sig/err);
    
    if  errdB > stopdB, %err < 1e-6,
        disp(['Stop iteration at err = ', num2str(errdB) ' dB']);
        break
    end
    
    disp(['iteration ' int2str(pp) ', error = ' num2str(err) ', dB = ', num2str(errdB)]);
    
    %  soundsc(real(curr),fs);
    %  dif = (abs(curr(1:length(signal))) - abs(signal')).^2;
    dm = (abs(y2)-y1).^2;
    %    dif = ifft(dm);
    Di = sum(sum(dm));  
   
    D(pp) = abs(Di);
end
rec = curr(1:end-window_length);

