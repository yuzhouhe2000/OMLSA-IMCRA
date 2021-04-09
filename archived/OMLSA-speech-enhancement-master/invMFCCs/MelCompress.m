function [yyout,mfcc]=MelCompress(Y,MelBankNum,fs)
[HalfFFTsize,FrameNum]=size(Y);
FFTsize = (HalfFFTsize-1)*2;
fs=8000;
NumLinFilts = MelBankNum/2; 
NumLogFilts = MelBankNum/2; 
frame_length = FFTsize;
lin_limit = 1000;
%% Create the Mel filterbank
[mel_filter,fc,a] = melbank(fs,NumLinFilts,NumLogFilts,frame_length,lin_limit);
bank = mel_filter;
%% calculate the MFCCs to each frame.
mfcc=zeros(MelBankNum,FrameNum);
yyout=zeros([HalfFFTsize,FrameNum]);
for i=1:FrameNum
    y=Y(:,i);
    y_am=y.^2; % compute the power spectrum
    melpow=bank*y_am; % the power spectrum is Mel-filtered
    mfcc(:,i) = dct(log(melpow));% compute the MFCCs
    melpow = exp(idct(mfcc(:,i)));
    %% use the LS algorithm to get the minimum norm solution,problem:some element of the solution vector is negative
    pow_rec = bank'*inv(bank*bank'+1e-9*speye(size(bank,1)))*melpow;
   %% ---Estimate the power spectrum using the iteratively reweighted L2 minimization method----%
    pow_rec=fIRLS(bank,melpow,1,20);
    AmSpectrum = abs(pow_rec).^0.5;%the constructed amplitute spectrum
   %% ---Estimate the spectral spectrum----%
    yyout(:,i) = AmSpectrum';
end
end

function xIRLS=fIRLS(A,y,p,Iternum)
xo=pinv(A)*y;
D=A;it=0;f=y;
res=1e10;tol=1e-7;itmax=Iternum;
while (res>tol & it<itmax)
   Wk=diag(abs(xo).^(1-p/2));
   xn=Wk*pinv(D*Wk)*f;
   res=norm(xn-xo);
   xo=xn;
   it=it+1;
end
xIRLS=xo;
end
