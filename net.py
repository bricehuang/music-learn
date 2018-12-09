import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import audioPreprocessor

class Net(nn.Module):
	def __init__(self, numClasses,include_dropout = True):
		super(Net, self).__init__()
		
		self.include_dropout = include_dropout

		self.channels = [1, 32, 64, 128]
		self.finalConv1 = nn.Conv2d(128, 256, kernel_size=3, padding=2)
		self.finalConv2 = nn.Conv2d(256, 256, kernel_size=3, padding=2)
		self.finalPool1 = nn.MaxPool2d(kernel_size=(7, 10))
		self.finalfc1 = nn.Linear(256, 1024)
		self.finalfc2 = nn.Linear(1024, numClasses)
		self.finaldrop = nn.Dropout(p=0.5)
		self.finalsig = nn.Sigmoid()
		self.finalleaky1 = nn.LeakyReLU(0.1)
		self.finalleaky2 = nn.LeakyReLU(0.1)

		self.conv1 = nn.Conv2d(1,32,kernel_size = 3, padding=2)
		self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=2)
		self.pool3 = nn.MaxPool2d(kernel_size=3)
		self.drop4 = nn.Dropout(p=0.25)
		self.conv5 = nn.Conv2d(32,64,kernel_size = 3, padding=2)
		self.conv6 = nn.Conv2d(64,64,kernel_size=3,padding=2)
		self.pool7 = nn.MaxPool2d(kernel_size=3)
		self.drop8 = nn.Dropout(p=0.25)
		self.conv9 = nn.Conv2d(64,128,kernel_size = 3, padding=2)
		self.conv10 = nn.Conv2d(128,128,kernel_size=3,padding=2)
		self.pool11 = nn.MaxPool2d(kernel_size=3)
		self.drop12 = nn.Dropout(p=0.25)

#		self.conv1 = torch.nn.ModuleList()
#		self.conv2 = torch.nn.ModuleList()
#		self.pool3 = torch.nn.ModuleList()
#		self.drop4 = torch.nn.ModuleList()
#		for i in range(len(self.channels) - 1):
#			self.conv1.append(nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=3, padding=2))
#			self.conv2.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=2))
#			self.pool3.append(nn.MaxPool2d(kernel_size=3))
#			self.drop4.append(nn.Dropout(p=0.25))


	def forward_block(self, x, ind):
		x = F.relu(self.conv1[ind](x))
		x = self.conv2[ind](x)
		x = self.pool3[ind](x)
		if self.include_dropout:
			x = self.drop4[ind](x)
		return x


	def forward(self, x):
		#print(x.shape)
#		for i in range(len(self.channels) - 1):
#			x = self.forward_block(x, i)

		x = F.relu(self.conv1(x))
		x = self.conv2(x)
		x = self.pool3(x)
		if self.include_dropout:
			x = self.drop4(x)
		x = F.relu(self.conv5(x))
		x = self.conv6(x)
		x = self.pool7(x)
		if self.include_dropout:
			x = self.drop8(x)
		x = F.relu(self.conv9(x))
		x = self.conv10(x)
		x = self.pool11(x)
		if self.include_dropout:
			x = self.drop12(x)

		x = F.relu(self.finalConv1(x))
		x = self.finalConv2(x)
		x = self.finalPool1(x)
		x = x.view(-1, 256)
		x = F.relu(self.finalfc1(x))
		if self.include_dropout:
			x = self.finaldrop(x)
		x = self.finalfc2(x)
		x = self.finalsig(x)
		return x

class ResNet(nn.Module):
	def __init__(self, numClasses):
		super(ResNet, self).__init__()

		self.channels = [1, 32, 64, 128]
		self.finalPool = nn.MaxPool2d(kernel_size=(1,4))
		self.finalfc1 = nn.Linear(128, 1024)
		self.finalfc2 = nn.Linear(1024, numClasses)
		self.finalsig = nn.Sigmoid()

		self.conv1 = nn.ModuleList()
		self.conv2 = nn.ModuleList()
		self.conv3 = nn.ModuleList()
		self.conv4 = nn.ModuleList()
		self.conv5 = nn.ModuleList()
		self.conv6 = nn.ModuleList()
		self.pool = nn.ModuleList()
		for i in range(len(self.channels) - 1):
			self.conv1.append(nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=3, padding=1))
			self.conv2.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1))
			self.conv3.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1))
			self.conv4.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1))
			self.conv5.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1))
			self.conv6.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1))

			self.pool.append(nn.MaxPool2d(kernel_size=3))


	def forward_block(self, x, ind):
		y = F.relu(self.conv1[ind](x))
		y = F.relu(self.conv2[ind](y))
		z = F.relu(self.conv3[ind](y))
		z = F.relu(self.conv4[ind](z) + y)
		w = F.relu(self.conv5[ind](z))
		w = F.relu(self.conv6[ind](w) + z)
		w = self.pool[ind](w)
		return w


	def forward(self, x):
		#print(x.shape)
		for i in range(len(self.channels) - 1):
			x = self.forward_block(x, i)
#			print(x.shape)
		x = self.finalPool(x)
		x = x.view(-1, 128)
		x = F.relu(self.finalfc1(x))
		x = self.finalfc2(x)
		x = self.finalsig(x)
		return x



def train(model, device, train_loader, optimizer, epoch):
	print("Training epoch" + str(epoch))
	print(next(model.parameters()).is_cuda)
	model.train()
	sum_num_correct = 0
	sum_loss = 0
	num_batches_since_log = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(device)
		target = target.to(device)
		print("Batch index " + str(batch_idx) + "/" + str(len(train_loader)))
		optimizer.zero_grad()
		#print(data[0])
#		print(next(model.parameters()).is_cuda)
#		print(data.is_cuda)
		output = model(data)
#		print(list(output))
#		print(target)
#		if (output < 1.0e-30).any():
#			print(batch_idx)
#			print(target)
#			exit()
		target = target.long().to(device)
		#print(output)
		#print(target)
		loss = F.cross_entropy(output, target)
		pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct = pred.eq(target.view_as(pred)).sum().item()
		sum_num_correct += correct
		sum_loss += loss.item()
		print(loss.item())
#		x = input("Enter to continue: ")
		num_batches_since_log += 1
		loss.backward()
		optimizer.step()
		if batch_idx + 1 == len(train_loader):
			print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
				epoch, batch_idx * len(data), len(train_loader),
				100. * batch_idx / len(train_loader), loss.item(),
				100. * sum_num_correct / (num_batches_since_log * 128))
			)
			return 100. * sum_num_correct / (num_batches_since_log * 128)
#			sum_num_correct = 0
#			sum_loss = 0
#			num_batches_since_log = 0

def test(model, device, data):
	model.eval()
	with torch.no_grad():
		data = data.to(device)
		output = model(data)
	return output