from pydub import AudioSegment
import librosa
from omlsa import omlsa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt

fs = 16000
input1 = "p287_004.wav"
input2 = "p287_005.wav"
input_dst = "input.wav"
out_dst = "out.wav"

y1, fs = librosa.load(input1, sr=fs)
y2, fs = librosa.load(input2, sr=fs)
white_noise = np.random.normal(0,0.15,len(y1))

# y_combine = y1 + y2[0:len(y1)]*0.1
y_combine = y1 + white_noise

# plt.plot(y_combine)
# plt.show()

write(input_dst,16000,y_combine) 

# choose between f (frequency domain plot), t (time domain plot), or None
output = omlsa(y_combine,fs, plot = "f")

write(out_dst,16000,output) 
print("done")