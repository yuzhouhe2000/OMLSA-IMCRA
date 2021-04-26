#FS = 44100
#F0 = CENTER FREQUENCY
#DBGAIN = FOR PEAKING AND SHELVING FILTERS, THE GAIN
#Q =  RELATED TO BANDWIDTH


#W0 = 2*PI *F0/FS
#COSW0 = cos(W0)
#SINW0 = sin(W0)
#alpha = SINW0/2*Q

#FOR PEAKING AND SHELVING EQ ONLY, CALCULATE
#A = 10^(DBGAIN/40)

import numpy as np
from scipy.io import wavfile
from scipy import signal
import math

def lowpass(w0, Q):
	#RETURNS SOS that can be used in SOSFILT
	#LPF
	#H(s) = 1/(s^2 + s/Q + 1)

	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = (1-COSW0)/2
	b1 = 1-COSW0
	b2 = b0
	a0 = 1 + alpha
	a1 = -2*COSW0
	a2 = 1 - alpha
	num = [b0, b1, b2]
	den = [a0, a1, a2]
	# sos = signal.tf2sos(num, den)

	# print(sos)
	return num,den

def highpass(w0, Q):
	#HPF
	#H(s) = (s^2) / ((s^2) + (s/Q) + 1)
	b0 = (1+COSW0)/2
	b1 = -(1+COSW0)
	b2 = b0
	a0 = 1 + alpha
	a1 = -2*COSW0
	a2 = 1 - alpha


def bandpass(w0, Q):
	#BPF using peak gain Q
	#H(s) = s / (s^2 + s/q + 1)
	b0 = Q*alpha #= SINW0/2
	b1 = 0
	b2 = -b0
	a0 = a + alpha
	a1 = -2*COSW0
	a2 = 1 - alpha


def peaking(w0, Q, A):
	#Peaking EQ
	#H(s) = (s^2 + s*A/Q + 1) / ( s^2 + s/(A*Q) + 1)
	b0 = 1 + (alpha*A)
	b1 = -2*COSW0
	b2 = 1 - (alpha*A)
	a0 = 1 + (alpha/A)
	a1 = -2*COSW0
	a2 = 1 - (alpha/A)


def lowShelf(w0, Q, A):
	#Low Shelf
	#H(s) = A * ((s^2 + sqrt(A)*s/Q + A) / (A*(s^2) + sqrt(A)*s/Q + 1)
	b0 = A*((A+1)-(A-1)*COSW0 + (2*sqrt(A)*alpha))
	b1 = 2*A*((A-1)-((A+1)*COSW0))
	b2 = A*((A+1)-(A-1)*COSW0 - (2*sqrt(A)*alpha))
	a0 = (A+1) + ((A-1)*COSW0) + (s*sqrt(A)*alpha)
	a1 = -2 * ((A-1) + ((A+1)*COSW0))
	a2 = (A+1) + ((A-1)*COSW0) - (2*sqrt(A)*alpha)


def highShelf(w0, Q, A):
	#High Shelf
	#H(s) = A * ((A*(s^2) + sqrt(A)*s/Q + 1) / ((s^2) + sqrt(A)*s/Q + A)
	b0 = A*((A+1)+(A-1)*COSW0 + (2*sqrt(A)*alpha))
	b1 = 2*A*((A-1)+((A+1)*COSW0))
	b2 = A*((A+1)+(A-1)*COSW0 - (2*sqrt(A)*alpha))
	a0 = (A+1) - ((A-1)*COSW0) + (s*sqrt(A)*alpha)
	a1 = -2 * ((A-1) - ((A+1)*COSW0))
	a2 = (A+1) - ((A-1)*COSW0) - (2*sqrt(A)*alpha)

def main():
	#SET
	FS = 44100

	#GET FROM USER
	F0 = 1000
	Q = 5
	#TYPE of filter, here use LPF

	#CALCULATE
	W0 = 2*math.pi*(F0/FS)
	num,den = lowpass(W0, Q)
	#READ IN AN AUDIO FILE to a np array inputArray
	sampleRate, inputArray = wavfile.read("test.wav")
	outputArray = signal.lfilter(num, den, inputArray)

	import matplotlib.pyplot as plt
	NFFT = 256
	fig, axes = plt.subplots(nrows=2, ncols=1)
	Pxx, freqs, bins, im = axes[0].specgram(inputArray,NFFT=NFFT, Fs=sampleRate, noverlap = NFFT/2)
	Pxx, freqs, bins, im = axes[1].specgram(outputArray,NFFT=NFFT, Fs=sampleRate, noverlap= NFFT/2)
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	plt.show()

	wavfile.write("processed.wav", sampleRate, outputArray)

if __name__ == '__main__':
	main()


