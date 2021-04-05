from pydub import AudioSegment
import librosa
from imcra import imcra

fs = 44100
dst = "in.wav"

y, fs = librosa.load(dst, sr=fs)

imcra(y,fs)