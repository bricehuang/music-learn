import scipy.io.wavfile as wf
import os
import numpy as np

def read(filename):
    _, data = wf.read(filename)
    return data

DATA_PATH = "./IRMAS-Sample"

def readTrainingAudio():
    dataList = []
    path = "./IRMAS-TrainingData"
    for label in os.listdir(path):
        if not os.path.isdir(path+"/"+label):
            continue
        for fname in os.listdir(path+"/"+label):
            data = read(path+"/"+label+"/"+fname)
            if len(data) < 3*44100:
                data = np.concatenate((data, np.zeros((3*44100 - len(data), 2), dtype="int16")))
            dataList.append((data,label))
    return dataList

def readTestAudio():
    dataList = []
    path = "./IRMAS-TestingData-Part1/Part1"
    for fname in os.listdir(path):
        fn, ext = os.path.splitext(fname)
        if ext == ".wav":
            tfile = fn + ".txt"
            pfh = open(path+"/"+tfile,"r")
            labels = pfh.read().split("\t\n")[:-1]
            data = read(path+"/"+fname)
            dataList.append((data,labels))
    return dataList
