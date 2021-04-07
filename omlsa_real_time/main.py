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
noise = np.random.normal(0,0.15,len(y1))

# y_combine = y1 + y2[0:len(y1)]*0.1
y_combine = y1 + noise

# plt.plot(y_combine)
# plt.show()

write(input_dst,16000,y_combine) 

frame_length = 512
frame_move = 128
# Turn plot on if you want compare spectrograms
output = omlsa(y_combine,fs,frame_length,frame_move)
write(out_dst,16000,output) 

print("done")