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
import matplotlib.pyplot as plt

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
	sos = signal.tf2sos(num, den)
	return sos

def highpass(w0, Q):
	#HPF
	#H(s) = (s^2) / ((s^2) + (s/Q) + 1)
	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = (1+COSW0)/2
	b1 = -(1+COSW0)
	b2 = b0
	a0 = 1 + alpha
	a1 = -2*COSW0
	a2 = 1 - alpha

	num = [b0, b1, b2]
	den = [a0, a1, a2]
	sos = signal.tf2sos(num, den)
	return sos

def bandpass(w0, Q):
	#BPF using peak gain Q
	#H(s) = s / (s^2 + s/q + 1)
	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = Q*alpha #= SINW0/2
	b1 = 0
	b2 = -b0
	a0 = a + alpha
	a1 = -2*COSW0
	a2 = 1 - alpha

	num = [b0, b1, b2]
	den = [a0, a1, a2]
	sos = signal.tf2sos(num, den)
	return sos


def peaking(w0, Q, A):
	#Peaking EQ
	#H(s) = (s^2 + s*A/Q + 1) / ( s^2 + s/(A*Q) + 1)
	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = 1 + (alpha*A)
	b1 = -2*COSW0
	b2 = 1 - (alpha*A)
	a0 = 1 + (alpha/A)
	a1 = -2*COSW0
	a2 = 1 - (alpha/A)

	num = [b0, b1, b2]
	den = [a0, a1, a2]
	sos = signal.tf2sos(num, den)
	return sos


def lowShelf(w0, Q, A):
	#Low Shelf
	#H(s) = A * ((s^2 + sqrt(A)*s/Q + A) / (A*(s^2) + sqrt(A)*s/Q + 1)
	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = A*((A+1)-(A-1)*COSW0 + (2*math.sqrt(A)*alpha))
	b1 = 2*A*((A-1)-((A+1)*COSW0))
	b2 = A*((A+1)-(A-1)*COSW0 - (2*math.sqrt(A)*alpha))
	a0 = (A+1) + ((A-1)*COSW0) + (2*math.sqrt(A)*alpha)
	a1 = -2 * ((A-1) + ((A+1)*COSW0))
	a2 = (A+1) + ((A-1)*COSW0) - (2*math.sqrt(A)*alpha)

	num = [b0, b1, b2]
	den = [a0, a1, a2]
	sos = signal.tf2sos(num, den)
	return sos


def highShelf(w0, Q, A):
	#High Shelf
	#H(s) = A * ((A*(s^2) + sqrt(A)*s/Q + 1) / ((s^2) + sqrt(A)*s/Q + A)
	COSW0 = math.cos(w0)
	alpha = math.sin(w0)/(2*Q)

	b0 = A*((A+1)+(A-1)*COSW0 + (2*math.sqrt(A)*alpha))
	b1 = 2*A*((A-1)+((A+1)*COSW0))
	b2 = A*((A+1)+(A-1)*COSW0 - (2*math.sqrt(A)*alpha))
	a0 = (A+1) - ((A-1)*COSW0) + (2*math.sqrt(A)*alpha)
	a1 = -2 * ((A-1) - ((A+1)*COSW0))
	a2 = (A+1) - ((A-1)*COSW0) - (2*math.sqrt(A)*alpha)

	num = [b0, b1, b2]
	den = [a0, a1, a2]
	sos = signal.tf2sos(num, den)
	return sos

def main():
	#SET
	FS = 44100
	sampleRate, inputArray = wavfile.read("test.wav")

	#GET FROM USER
	print("LPF================================")
	F0 = 4000
	Q = 3
	W0 = 2*math.pi*(F0/FS)
	sos1 = lowpass(W0, Q)
	inputArray2 = signal.sosfilt(sos1, inputArray)

	print("Low Shelf================================")
	F0 = 3000
	Q = 3
	A = 1
	W0 = 2*math.pi*(F0/FS)
	sos2 = lowShelf(W0, Q, A)
	inputArray3 = signal.sosfilt(sos2, inputArray2)

	print("Peaking EQ================================")
	F0 = 1000
	Q = 3
	A = 1
	W0 = 2*math.pi*(F0/FS)
	sos3 = peaking(W0, Q, A)
	inputArray4 = signal.sosfilt(sos3, inputArray3)

	print("High Shelf================================")
	F0 = 250
	Q = 3
	A = 1
	W0 = 2*math.pi*(F0/FS)
	sos4 = highShelf(W0, Q, A)
	inputArray5 = signal.sosfilt(sos4, inputArray4)

	print("HPF================================")
	F0 = 20
	Q = 3
	W0 = 2*math.pi*(F0/FS)
	sos5 = highpass(W0, Q)
	outputArray = signal.sosfilt(sos5, inputArray5)



	w, h1 = signal.sosfreqz(sos1)
	w, h2 = signal.sosfreqz(sos2)
	w, h3 = signal.sosfreqz(sos3)
	w, h4 = signal.sosfreqz(sos4)
	w, h5 = signal.sosfreqz(sos5)

	print(sos3.shape)
	print(sos2)
	# db = 20 * np.log10(np.maximum(np.abs(h1*h2*h3*h3*h4*h5), 1e-5))
	# plt.plot(w/np.pi, db)
	# plt.show()

	# #CALCULATE
	# #READ IN AN AUDIO FILE to a np array inputArray
	# wavfile.write("processed.wav", sampleRate, outputArray)

if __name__ == '__main__':
	main()
