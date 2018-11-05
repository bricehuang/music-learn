import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import audioPreprocessor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels = [1, 32, 64, 128]
        self.finalConv1 = nn.Conv2d(128, 256, kernel_size=3, padding=2)
        self.finalConv2 = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.finalPool1 = nn.MaxPool2d(kernel_size=(7, 10))
        self.finalfc1 = nn.Linear(256, 1024)
        self.finalfc2 = nn.Linear(1024, 11)
        self.finaldrop = nn.Dropout(p=0.5)
        self.finalsig = nn.Sigmoid()

        self.conv1 = []
        self.conv2 = []
        self.pool3 = []
        self.drop4 = []
        for i in range(len(self.channels) - 1):
            self.conv1.append(nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=3, padding=2))
            self.conv2.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=2))
            self.pool3.append(nn.MaxPool2d(kernel_size=3))
            self.drop4.append(nn.Dropout(p=0.25))


    def forward_block(self, x, ind):
        x = F.relu(self.conv1[ind](x))
        x = F.relu(self.conv2[ind](x))
        x = self.pool3[ind](x)
        x = self.drop4[ind](x)
        return x


    def forward(self, x):
        #print(x.shape)
        for i in range(len(self.channels) - 1):
            x = self.forward_block(x, i)
        x = F.relu(self.finalConv1(x))
        x = F.relu(self.finalConv2(x))
        x = self.finalPool1(x)
        #print(x.shape)
        x = x.view(-1, 256)
        x = F.relu(self.finalfc1(x))
        x = self.finaldrop(x)
        x = F.relu(self.finalfc2(x))
        x = self.finalsig(x)
        return x


def train(model, train_loader, optimizer, epoch):
    print("Training epoch" + str(epoch))
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Batch index " + str(batch_idx) + "/" + str(len(train_loader)))
        optimizer.zero_grad()
        #print(data[0])
        output = model(data)
        target = target.long()
        #print(output)
        #print(target)
        loss = F.cross_entropy(output, target)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item()
        print(loss.item())
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx + 1 == len(train_loader):
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * sum_num_correct / (num_batches_since_log * 128))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0

def test(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output