from .scan import Scan
import numpy as np

class MontecarloScan(Scan):
	def __init__(self, likelihood, par_min, par_max, N_iters, bf, varsize=0.25, *args):
		Scan.__init__(self, likelihood, par_min, par_max, N_iters)
		self.bf = np.array(bf)
		self.lh = self.likelihood(self.bf, *args)
		self.varsize = varsize
	def run(self, *args):
		self.variances = np.zeros(self.Npars)
		for p in range(0, self.Npars):
			self.variances[p] = (self.par_max[p]-self.par_min[p])*self.varsize					
		N = 0
		Ntot = 0
		while N < self.N_iters:
			valid = False
			while not valid:
				p0 = self.bf + np.random.randn(self.Npars) * self.variances
				valid = self.inthebox(p0)
			lh0 = self.likelihood(p0, *args)
			if lh0 > self.lh + np.log(np.random.uniform()): #Acceptance condition for log-likelihoods!!!
				self.points.append(p0)
				self.lh_list.append(lh0)
				N += 1
			Ntot +=1
		self.increasecounter(Ntot)

				


