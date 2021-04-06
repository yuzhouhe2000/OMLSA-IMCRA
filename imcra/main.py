from pydub import AudioSegment
import librosa
from imcra import imcra
from scipy.io.wavfile import write

fs = 16000
dst = "in.wav"
out_dst = "out.wav"

y, fs = librosa.load(dst, sr=fs)

output = imcra(y,fs)

write(out_dst,16000,output) 
print("done")