function [a] =  istftw2 (B,hopfactor,window)
%this function calculate the inverse STFT for a STFT matrix
% B - STFT matrix
% hopfactor - number of analysis hops occurring within a frame
% window - window values
% (c) Shlomo Dubnov sdubnov@ucsd.edu

B = [B; conj(B(end-1:-1:2,:))];


nfft = length(window);
bmat = ifft(B);
step = nfft/hopfactor;

w2 = window .* window;

[M N] = size(bmat);

a = zeros (1,N*step + nfft);
w2sum = zeros (1,N*step + nfft);
win_pos = [1: step: length(a) - nfft];

for i=1:length(win_pos)
   a(win_pos(i):win_pos(i)+nfft-1) = a(win_pos(i):win_pos(i)+nfft-1) + (bmat(:,i).').*(window.');
   w2sum(win_pos(i):win_pos(i)+nfft-1) = w2sum(win_pos(i):win_pos(i)+nfft-1) + w2'; 
end

a(step:end-nfft) = a(step:end-nfft)./w2sum(step:end-nfft);
a = real(a);
