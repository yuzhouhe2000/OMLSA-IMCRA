import numpy as np
import scipy


def reformat(input):
    input = input.reshape(len(input),1)
    return input

def clipper(a,b,command):
    if command == "max":
        for i in range(0, len(a)):
            if (a[i] < b):
                a[i] = b
    if command == "min":
        for i in range(0, len(a)):
            if (a[i] > b):
                a[i] = b
    return a

def element_wise_power(a,b):
    output = []
    for i in range(0,len(a)):
        output.append(np.power(a[i],b[i]))
    return output

def reformat_ifft(input):
    output = []
    for i in input:
        output.append(i[0])
    return np.array(output)

def Reverse(lst):
    return [ele for ele in reversed(lst)]

def find_nonzero(input):
    output = []
    for i in range(0,len(input)):
        if input[i] != 0:
            output.append(i)
    return output

def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

def circular_shift(x,t):
    return [x[t:len(x)], x[0:t]]

