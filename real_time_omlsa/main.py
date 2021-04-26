from scipy.io.wavfile import read
from omlsa import *
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sounddevice as sd
import threading
import time
import sys


buffer = []
LIVE = 1
sample_rate = 44100
frame_length = 256
frame_move = 128

def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        sys.exit(1)
    return caps


def audio_input():
    global buffer 
    global LIVE  

    caps = query_devices(None, "input")
    channels_in = min(caps['max_input_channels'], 1)
    stream_in = sd.InputStream(
        device=None,
        samplerate=sample_rate,
        channels=channels_in)
    stream_in.start()
    while (1):
        frame, overflow = stream_in.read(frame_move)
        buffer.append(frame)
        
    stream_in.stop()
    return ('', 204)

def denoiser_output():
    global LIVE
    device_out = "Soundflower (2ch)"
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 1)
    stream_out = sd.OutputStream(
        device=None,
        samplerate=sample_rate,
        channels=channels_out)
    stream_out.start()
    while(1):
        while (LIVE == 1):
            if buffer != []:
                while len(buffer) > 10:
                    del(buffer[0])
                frame = buffer[0]
                del(buffer[0])
                # print(len(buffer))
                start = time.time()
                output = omlsa_streamer(frame,sample_rate, frame_length, frame_move,postprocess= "butter",high_cut=19000)
                print(time.time()-start)
                stream_out.write(output.astype(np.float32))
        while(LIVE == 0):
            if buffer != []:
                while len(buffer) > 10:
                    del(buffer[0])
                frame = buffer[0]
                del(buffer[0])      
                stream_out.write(frame)
    stream_out.stop()

def switch():
    global LIVE
    while(1):
        if LIVE == 1:
            input("Press Enter to continue...")
            print("denoiser_off")
            LIVE = 0
        else:
            input("Press Enter to continue...")
            print("denoiser_on")
            LIVE = 1

threads = []
threads.append(threading.Thread(target=audio_input))
threads.append(threading.Thread(target=denoiser_output))
threads.append(threading.Thread(target=switch))
print(threads)

if __name__ == '__main__':
    for thread in threads:
        thread.start()