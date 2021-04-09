from scipy.io.wavfile import read
from omlsa import omlsa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

input1 = "p287_004.wav"
input2 = "p287_005.wav"
input_dst = "input.wav"
out_dst = "out.wav"

fs, y1 = read(input1)
y1 = (y1 / 32767).astype(np.float)
# fs,y2 = read(input2)
# y2 = (y2 / 32767).astype(np.float)

# y1 = scipy.signal.resample(y1, int(len(y1)/16000*44100))
white_noise = np.random.normal(0,0.0000000001,len(y1))
# y_combine = y1 + y2[0:len(y1)]*0.6
y_combine = y1 + white_noise
y_combine = np.repeat(y_combine,10)
write(input_dst,fs,y_combine)
# choose between f (frequency domain plot), t (time domain plot), or None
# can also set up high cut, default is 15000
output = omlsa(y_combine,fs, frame_length = 512, frame_move = 256, plot = "f",preprocess = None)
write(out_dst,fs,output) 
print("done")