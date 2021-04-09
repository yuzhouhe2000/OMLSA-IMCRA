import torch
from torch import nn
import os
import math
import sys
import numpy as np
import python_speech_features
import array
from torch.nn import Linear, RNN, LSTM, GRU
import torch.nn.functional as F
from torch.nn.functional import softmax, relu
from torch.autograd import Variable
import pyaudio
import numpy
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001


# Define Global Variable
LIVE = 1
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2
BATCH_SIZE = 1
# Frame size to use for the labelling.
FRAME_SIZE_MS = 30
# Calculate frame size in data points.
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))
FRAMES = 20
FEATURES = 12
OBJ_CUDA = torch.cuda.is_available()

if OBJ_CUDA:
    print('CUDA has been enabled.')
else:
    print('CUDA has been disabled.')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.hidden = self.init_hidden()
        self.rnn = LSTM(input_size=FEATURES, hidden_size=FRAMES, num_layers=1, batch_first=True)
        self.lin1 = nn.Linear(FRAMES**2, 26)
        self.lin2 = nn.Linear(26, 2)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = Variable(torch.zeros(1, BATCH_SIZE, FRAMES))
        c = Variable(torch.zeros(1, BATCH_SIZE, FRAMES))
        if OBJ_CUDA:
            h = h.cuda()
            c = c.cuda()
        return h, c
    def forward(self, x):
        x, _ = self.rnn(x, self.hidden)

        x = x.contiguous().view(-1, FRAMES**2)

        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        
        return self.softmax(x)

model = Net()
print(model)
            
RESULT = ""

mfcc_stream = []
count = 0
result = -1
model.load_state_dict(torch.load("lstm.net",map_location=torch.device(device)))
model.eval()

def denoiser_VAD(frame):
    global result
    global mfcc_stream
    global count
    
    frame = frame.reshape((-1,)) 
    length = int(len(frame)/FRAME_SIZE)
    # print(FRAME_SIZE)
    
 
    frame_list = frame[0:FRAME_SIZE*length]
    
    for frame_n in range(0,length):
        frame = frame_list[frame_n*FRAME_SIZE:(frame_n+1)*FRAME_SIZE]
        with torch.no_grad():
            mfcc_feature = python_speech_features.mfcc(frame,SAMPLE_RATE, winstep=(FRAME_SIZE_MS / 1000), 
                                        winlen= 4 * (FRAME_SIZE_MS / 1000), nfft=2048)
            mfcc_feature = mfcc_feature[:, 1:]

            mfcc_stream.append(mfcc_feature)
            if len(mfcc_stream) >= (FRAMES+1):
                mfcc_stream.pop(0)
            count = count + 1

            # print(len(mfcc_stream))

            if len(mfcc_stream) == FRAMES:
                
                if count >=1:
                    mfcc_stream_array = np.array(mfcc_stream)
                    mfcc_stream_array = mfcc_stream_array.transpose(1,0,2)
                    mfcc_stream_array = mfcc_stream_array.astype(np.float32)
                    test_data_tensor = torch.from_numpy(mfcc_stream_array).to(device)
                    output = model(test_data_tensor).to(device)
                    pred_y = torch.max(output, 1)[1].data.numpy()
                    result = pred_y[0]
                    end = time.time()
                    count = 0
                    print(result)
                    return(result)
    return(result)


