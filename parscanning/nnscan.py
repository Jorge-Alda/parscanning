from .scan import Scan
from .NN import NN
from .randomscan import RandomScan
from multiprocessing import Pool
import numpy as np
from itertools import product
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class NNScan(Scan):
	def __init__(self, likelihood, par_min, par_max, N_iters, bf, cutoff=None):
		Scan.__init__(self, likelihood, par_min, par_max, N_iters)
		self.bf = np.array(bf)
		self.lh = self.likelihood(self.bf)
		self.cutoff = cutoff

	def init_NN(self, neurons, LR=0.001, epochs=5000, batch_size = 128, verbose=False, varsize=0.5):
		self.NN = NN(neurons, self.Npars, 1, LR, epochs, batch_size, verbose)
		self.started_class = False
		self.started_pred = False
		self.variances = np.zeros(self.Npars)
		for p in range(0, self.Npars):
			self.variances[p] = (self.par_max[p]-self.par_min[p])*varsize

	def run(self):
		self.run_mp(1)

	def newpoints(self, num=None):
		if num is None:
			num = self.N_iters
		if 'NN' not in self.__dict__.keys():
			raise NotImplementedError('You have to init the Neural Network first!')

		if self.started_pred:
			xlist = []
			Ntot = 0
			while len(xlist) < num:
				Ntot += 1
				x0 = self.bf + np.random.randn(self.Npars) * self.variances
				if self.started_class:
					if not self.guess_class(x0, self.cutoff):
						continue
				if self.guess_lh(x0) > self.lh + np.log(np.random.uniform()):
					xlist.append(x0)
		else:
			xlist = []
			Ntot = 0
			if self.started_class:
				while len(xlist) < num:
					Ntot += 1
					x0 = self.bf + np.random.randn(self.Npars) * self.variances
					if self.guess_class(x0, self.cutoff):
						xlist.append(x0)
			else:
				raise NotImplementedError('You have to train or load the Neural Network first!')

		self.Ntot += Ntot
		return xlist

	def run_mp(self, cores):
		xlist = self.newpoints()
		if cores == 1:
			lhlist = list(map(self.likelihood, xlist))
		else:
			self.mp = True
			with Pool(processes=cores) as pool:
				lhlist = pool.map(self.likelihood, xlist)
		self.points += xlist
		self.lh_list += lhlist	

	def train_class(self, x_val, y_val, cores=1):
		self.NN.set_classifier()
		ymax = max(y_val)
		ychi = 2*(ymax-np.array(y_val))
		sf = chi2.sf(ychi, self.Npars)
		classes = [(s, 1-s) for s in sf]
		self.NN.train(x_val, classes, cores, mode='class')
		self.started_class = True

	def train_pred(self, x_val, y_val, cores=1):
		self.NN.set_predictor()
		self.NN.train(x_val, y_val, cores, mode='pred')
		self.started_pred = True

	def guess_scoreclass(self, x):
		return self.NN.classifier(Variable(torch.FloatTensor(x))).data.cpu().numpy()[0]

	def guess_class(self, x, cutoff=0.5):
		yclass = self.NN.classifier(Variable(torch.FloatTensor(x))).data.cpu().numpy()
		return yclass[0] > cutoff

	def guess_lh(self, x):
		return float(self.NN.predictor(Variable(torch.FloatTensor(x))).data.cpu())

	def saveNN(self, fname, mode='pred'):
		if mode == 'pred':
			torch.save(self.NN.predictor.state_dict(), fname)
		elif mode == 'class':
			torch.save(self.NN.classifier.state_dict(), fname)

	def loadNN(self, fname, mode='pred'):
		if mode == 'pred':
			self.NN.set_predictor()
			self.NN.predictor.load_state_dict(torch.load(fname))
			self.started_pred = True
		elif mode == 'class':
			self.NN.set_classifier()
			self.NN.classifier.load_state_dict(torch.load(fname))
			self.started_class = True

