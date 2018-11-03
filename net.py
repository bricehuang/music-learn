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
        self.finalConv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.finalConv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward_block(self, x, inChannels, outChannels):
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3)
        self.drop4 = nn.Dropout(p=0.25)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = self.drop4(x)
        return x


    def forward(self, x):
        for i in range(len(channels) - 1):
            x = self.forward_block(x, channels[i], channels[i+1])
        x = F.relu(self.finalConv1(x))
        x = F.relu(self.finalConv2(x))
        x = nn.MaxPool2d(kernel_size=x.shape())(x)
        x = x.view(-1, 256)
        x = nn.Linear(256,1024)(x)
        x = nn.Dropout(p=0.5)(x)
        x = nn.Linear(1024, 11)(x)
        x = nn.Sigmoid()(x)

        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0

def test(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output