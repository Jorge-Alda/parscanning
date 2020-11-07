import numpy as np
import random
import tempfile
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# -------------------------------------
# the basic class for neural networks
# Implementation by F. Staub in https://github.com/fstaub/xBIT
# -------------------------------------
class NN():
	def __init__(self, neurons, input_dim, output_dim=1, LR=0.001, epochs=5000, batch_size=128, verbose=False):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.epochs = epochs
		self.neurons = neurons
		self.LR = LR
		self.batch_size = batch_size
		self.verbose = verbose
		self.tempdir = tempfile.mkdtemp()
		self.fout = os.path.join(self.tempdir, 'state.pth.tar')
		if self.verbose:
			print('Temporal files: ' + self.fout)

	#def __del__(self):
	#	shutil.rmtree(self.tempdir)

	def set_predictor(self):
		''' network which predicts numerical values for observables'''
		layers = [nn.Linear(self.input_dim, self.neurons[0]), nn.ReLU(), nn.Dropout(0.1)]
		for i in range(1, len(self.neurons)):
			layers.append(nn.Linear(self.neurons[i - 1], self.neurons[i]))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(0.1))
		layers.append(nn.Linear(self.neurons[-1], self.output_dim))
		self.predictor = nn.Sequential(*layers)

		self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=self.LR)
		self.predictor_criterion = nn.MSELoss()

	def set_classifier(self):
		''' network which checks if a point is valid or not'''
		self.classifier = nn.Sequential(nn.Linear(self.input_dim, 50), nn.ReLU(), nn.Dropout(0.1), nn.Linear(50,50), nn.ReLU(), nn.Dropout(0.1), nn.Linear(50, 2), nn.Sigmoid())
		self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=self.LR)
		self.classifier_criterion = nn.BCELoss()

	# training the neural network
	def train(self, x_val, y_val, cores, mode='pred'):
		torch.set_num_threads(cores)

		best = float('Inf')
		wait = 0
		
		nr_batches = max(1, int(len(x_val) / self.batch_size))
		patience = max(200, self.epochs / 2 / nr_batches)

		if mode == 'pred':
			training = self.predictor
			criterion = self.predictor_criterion
			optimizer = self.predictor_optimizer
			if self.verbose:
				print('Training predictor NNetwork')
		elif mode == 'class':
			training = self.classifier
			criterion = self.classifier_criterion
			optimizer = self.classifier_optimizer
			if self.verbose:
				print('Training classifier NNetwork')

		rand = list(zip(x_val, y_val))
		random.shuffle(rand)
		x_r, y_r = zip(*rand)

		x_train = torch.FloatTensor(x_r[:int(0.8 * len(x_r))])
		y_train = torch.FloatTensor(y_r[:int(0.8 * len(x_r))])

		x_test = torch.FloatTensor(x_r[int(0.8 * len(x_r)) + 1:])
		y_test = torch.FloatTensor(y_r[int(0.8 * len(x_r)) + 1:])

		# for mini batch
		# permutation = torch.randperm(len(x_train))
		permutation = torch.randperm(x_train.size()[0])

		for epoch in range(self.epochs):
			training.train()

			# mini batches
			for i in range(0, x_train.size()[0], self.batch_size):
				optimizer.zero_grad()
				indices = permutation[i:i + self.batch_size]
				batch_x, batch_y = x_train[indices], y_train[indices]
				y = training(Variable(batch_x))
				if mode == 'pred':
					y = y.flatten()
				self.loss = criterion(y, Variable(batch_y))

				self.loss.backward()
				optimizer.step()

			# calculate loss with test data
			training.eval()
			y_pred = training(Variable(x_test)).data.cpu()
			y_pred = y_pred.flatten()
			loss = criterion(Variable(y_pred), Variable(y_test))

			if epoch % 100 == 0 and self.verbose:
				print("Epoch: %i;  loss: %f" % (epoch, loss))

			# Simple implementation of early stopping
			  
			if loss < best:
				best = loss
				wait = 0
				torch.save({'state_dict': training.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': self.loss}, self.fout)
			else:
				wait = wait + 0.2
			if wait > patience or epoch == (self.epochs - 1):
				if self.verbose:
					print("Stopped after %i epochs" % epoch)
				checkpoint = torch.load(self.fout)
				training.load_state_dict(checkpoint['state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer'])
				self.loss = checkpoint['loss']
				break
			
			
		if self.verbose:
			print("Training Data: %i points, loss: %f" % (len(x_val), best))


