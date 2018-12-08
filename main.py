import audioRead as ar
import audioPreprocessor as ap
import net

import pickle

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet as resnet

import stats

import random

SNIPPET_WINDOW = int(1.0*44100)
SNIPPET_HOP = int(SNIPPET_WINDOW/2)
DOWNSAMPLE_RATE = 0.5
FFT_WINDOW = 1024
FFT_HOP = 512
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_BATCH = 128

THRESHOLD = 0.1

instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = range(11)
VALIDATION_FRAC = 0.15

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
#    print(torch.Tensor(snippet).shape)
#    snippet = ap.melSpectrogram(snippet)
#    snippet = ap.logCompress(snippet)
    return snippet

def batchify(procData, batchsize):
    #random.shuffle(procData)
    L = list(zip(*procData))
    M = []
    for iStart in range(0, len(procData), batchsize):
        obj = []
        for i in range(len(L)):
            tens = torch.Tensor(L[i][iStart:iStart+batchsize])
            if i == 0:  # data needs to have a "channel" dimension
                tens = tens.unsqueeze(1)
            obj.append(tens)
        M.append(obj)
    return M

try:
    pfh = open('procTrainingData', 'rb')
    procTrainingData = pickle.load(pfh)
    pfh.close()
    print("Loaded already-processed training data")
except:
    rawTrainingData = ar.readTrainingAudio()
    random.shuffle(rawTrainingData)
    rawTrainingData = rawTrainingData

    print("Finished reading data")

    procTrainingData = []

    counter = 0
    for (clip, label) in rawTrainingData:
        snippets = ap.divide(clip, SNIPPET_WINDOW, SNIPPET_WINDOW)
        labelvec = encodeLabels(label)
        for snippet in snippets:
            snippet = processSnippet(snippet)
            procTrainingData.append((snippet, instruments.index(label)))
        counter += 1
        if counter % 10 == 0:
            print("Processed " + str(counter) + "/" + str(len(rawTrainingData)) + " training clips")
    pfh = open('procTrainingDataUmel', 'wb')
    pickle.dump(procTrainingData, pfh, pickle.HIGHEST_PROTOCOL)
    pfh.close()

#random.shuffle(procTrainingData)
random.Random(4).shuffle(procTrainingData)
procTrainingData = procTrainingData[:2998+128*112] + procTrainingData[2998+128*113:]

filteredTrainingData = []
count = [0]*11
for data in procTrainingData:
    if data[1] in CLASSES:
#        print(data[1])
        data = (data[0], CLASSES.index(data[1]))
        filteredTrainingData.append(data)
        count[data[1]] += 1
#        plt.imshow(data[0], cmap='hot', interpolation='nearest')
#        plt.show()

#classSize = min(count)
#cutFilteredTrainingData = []
#for data in filteredTrainingData:
#    if random.randint(1, count[data[1]]) <= classSize:
#    if random.randint(1,sum(count)) <= len(CLASSES)*classSize:
#        cutFilteredTrainingData.append(data)
#filteredTrainingData = cutFilteredTrainingData

#random.shuffle(filteredTrainingData)
N = len(filteredTrainingData)
V = int(N*VALIDATION_FRAC)
filteredValidationData = filteredTrainingData[:V]
filteredTrainingData = filteredTrainingData[V:]

batchTrainingData = batchify(filteredTrainingData, TRAIN_BATCH)
batchValidationData = batchify(filteredValidationData, TRAIN_BATCH)

print("Batchified training data")

torch.manual_seed(1)

model = net.Net(len(CLASSES)).to(device)
print(next(model.parameters()).is_cuda)
#model = resnet.ResNet(resnet.BasicBlock,[2,2,2,2],num_classes=11)
#model.conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

lTrainAcc = []
lTestAcc = []
lF1 = []

for epoch in range(1, 100):
    trainAcc = net.train(model, device, batchTrainingData, optimizer, epoch)
    lTrainAcc.append(trainAcc)
    totalCorrect = 0
    total = 0
    dist = np.zeros((11,11))
    countTargets = np.zeros((11))
    for batch in batchValidationData:
        data = batch[0].to(device)
        target = batch[1].to(device)
        output = net.test(model, device, data)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        target = target.long().view_as(pred)
        totalCorrect += pred.eq(target).sum().item()
        total += len(data)
        for i in range(len(pred)):
            dist[target[i]][pred[i]] += 1
            countTargets[target[i]] += 1
    print("Test accuracy:" + str(100*totalCorrect/total) + "%")
    print(dist)
    lTestAcc.append(100*totalCorrect/total)
    lF1.append(stats.F1overall(np.array(dist)))
    print(lTrainAcc)
    print(lTestAcc)
    print(lF1)
#    woah = input("Enter  to continue...")


'''
rawTestData = ar.readTestAudio()

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
'''