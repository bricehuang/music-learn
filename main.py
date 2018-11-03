import audioRead as ar
import audioPreprocessor as ap
import net


import torch
import torch.nn.functional as F

import random

SNIPPET_WINDOW = 0.5*22050
SNIPPET_HOP = SNIPPET_WINDOW/2
DOWNSAMPLE_RATE = 0.5
FFT_WINDOW = 1024
FFT_HOP = 512
EPOCHS = 10
LR = 0.01

TRAIN_BATCH = 64

THRESHOLD = 0.1

instruments = ["cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

def encodeLabels(labels):
    if not type(labels) == list:
        labels = [labels]
    vec = [0]*len(instruments)
    for label in labels:
        vec[instruments.index(label)] = 1
    return vec

def processSnippet(snippet):
    snippet = ap.toMono(snippet)
    snippet = ap.downsample(snippet, DOWNSAMPLE_RATE)
    snippet = ap.normalize(snippet)
    snippet = ap.stft(snippet, FFT_WINDOW, FFT_HOP)
    snippet = ap.melSpectrogram(snippet)
    snippet = ap.logCompress(snippet)
    return snippet

def batchify(procData, batchsize):
    random.shuffle(procData)
    L = zip(*procData)
    M = []
    for iStart in range(0, len(procData), batchsize):
        obj = []
        for i in range(len(L)):
            obj.append(torch.Tensor(L[i][iStart:iStart+batchsize]))
        M.append(obj)
    return M

def divide(clip, window, hop):
    L = []
    for i in range(0,len(clip),hop):
        L.append(clip[i:i+window])
    return L

rawTrainingData = ar.readTrainingAudio()
rawTestData = ap.readTestAudio()

procTrainingData = []

for (clip, label) in rawTrainingData:
    snippets = ap.divide(clip, SNIPPET_WINDOW, SNIPPET_WINDOW)
    labelvec = encodeLabels(label)
    for snippet in snippets:
        snippet = processSnippet(snippet)
        procTrainingData.append((snippet, labelvec))

procTrainingData = batchify(procTrainingData, TRAIN_BATCH)

model = net.Net()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    net.train(model, procTrainingData, optimizer, epoch)

procTestData = []

loss = 0

for i, (clip, labels) in enumerate(rawTestData):
    snippets = ap.divide(clip, SNIPPET_WINDOW, SNIPPET_HOP)
    labelvec = encodeLabels(label)
    testData = []
    for snippet in snippets:
        snippet = processSnippet(snippet)
        testData.append(snippet)
    testData = torch.Tensor(testData)
    output = net.test(model, testData)
    output = torch.mean(output, 0)
    output = (output >= THRESHOLD)
    loss += abs(output - labelvec)

print(loss/len(rawTestData))
