from .scan import Scan
import numpy as np

class MontecarloScan(Scan):
	def __init__(self, likelihood, par_min, par_max, N_iters, bf):
		Scan.__init__(self, likelihood, par_min, par_max, N_iters)
		self.bf = np.array(bf)
		self.lh = self.likelihood(self.bf)
	def run(self):
		self.variances = np.zeros(self.Npars)
		for p in range(0, self.Npars):
			self.variances[p] = (self.par_max[p]-self.par_min[p])/4					
		N = 0
		while N < self.N_iters:
			valid = False
			while not valid:
				p0 = self.bf + np.random.randn(self.Npars) * self.variances
				valid = self.inthebox(p0)
			lh0 = self.likelihood(p0)
			if lh0 > self.lh + np.log(np.random.uniform()): #Acceptance condition for log-likelihoods!!!
				self.points.append(p0)
				self.lh_list.append(lh0)
				N += 1
			self.Ntot +=1

				


