import audioRead as ar
import audioPreprocessor as ap
import net
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

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

CLASSES = [6, 9, 10]
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
    snippet = ap.melSpectrogram(snippet)
    snippet = ap.logCompress(snippet)
    return snippet

def batchify(procData, batchsize):
    random.shuffle(procData)
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
    rawTrainingData = rawTrainingData[:25]

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
    pfh = open('procTrainingData', 'wb')
    pickle.dump(procTrainingData, pfh, pickle.HIGHEST_PROTOCOL)
    pfh.close()

filteredTrainingData = []
count = [0]*11
for data in procTrainingData:
    if data[1] in CLASSES:
        data = (data[0], CLASSES.index(data[1]))
        filteredTrainingData.append(data)

random.shuffle(filteredTrainingData)
N = len(filteredTrainingData)
V = int(N*VALIDATION_FRAC)
filteredValidationData = filteredTrainingData[:V]
filteredTrainingData = filteredTrainingData[V:]


batchTrainingData = batchify(filteredTrainingData, TRAIN_BATCH)
batchValidationData = batchify(filteredValidationData, TRAIN_BATCH)

print("Batchified training data")

torch.manual_seed(1)

model = net.Net(len(CLASSES))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, 1000 + 1):
    net.train(model, batchTrainingData, optimizer, epoch)
    totalCorrect = 0
    total = 0
    for batch in batchValidationData:
        output = net.test(model, batch[0])
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        totalCorrect += pred.eq(batch[1].long().view_as(pred)).sum().item()
        total += len(batch[0])
    print("Test accuracy:" + str(100*totalCorrect/total) + "%")


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