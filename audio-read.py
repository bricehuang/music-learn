import scipy.io.wavfile as wf
import os

def read(filename):
    _, data = wf.read(filename)
    return data

DATA_PATH = "./IRMAS-Sample"

def readTrainingAudio():
    dataList = []
    path = DATA_PATH + "/Training"
    for label in os.listdir(path):
        for fname in os.listdir(path+"/"+label):
            data = read(path+"/"+label+"/"+fname)
            dataList.append((data,label))
    return dataList

def readTestAudio():
    dataList = []
    path = DATA_PATH + "/Testing"
    for fname in os.listdir(path):
        fn, ext = os.path.splitext(fname)
        if ext == ".wav":
            tfile = fn + ".txt"
            pfh = open(path+"/"+tfile,"r")
            labels = pfh.read().split("\t\n")[:-1]
            data = read(path+"/"+fname)
            dataList.append((data,labels))
    return dataList

