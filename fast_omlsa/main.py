from scipy.io.wavfile import read
from omlsa import omlsa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt


input1 = "p287_004.wav"
input2 = "p287_005.wav"
input_dst = "input.wav"
out_dst = "out.wav"


fs, y1 = read(input1)

y1 = (y1 / 32767).astype(np.float)

fs,y2 = read(input2)
y2 = (y2 / 32767).astype(np.float)
white_noise = np.random.normal(0,0.02,len(y1))

# y_combine = y1 + y2[0:len(y1)]*0.1
y_combine = y1 + white_noise

# plt.plot(y_combine)
# plt.show()

write(input_dst,fs,y_combine) 

# choose between f (frequency domain plot), t (time domain plot), or None
output = omlsa(y_combine,fs, plot = "f")

write(out_dst,fs,output) 
print("done")