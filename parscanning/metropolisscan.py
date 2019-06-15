from .scan import Scan
import numpy as np

class MetropolisScan(Scan):
	def run(self, *args):
		if isinstance(self.N_iters, int):
			Nburns = int(self.N_iters/5)
			Nvalid = self.N_iters
		else:
			Nburns = self.N_iters[0]
			Nvalid = self.N_iters[1]
		if len(self.points) == 0: 
			self.variances = np.zeros(self.Npars)
			self.point = np.zeros(self.Npars)
			for p in range(0, self.Npars):
				self.point[p] = (self.par_max[p]-self.par_min[p])*np.random.random() + self.par_min[p]
				self.variances[p] = (self.par_max[p]-self.par_min[p])/(5*Nvalid**(1/self.Npars)	)		
			self.lh = self.likelihood(self.point, *args)
			for _ in range(0, Nburns):
				self._newpoint()
			self.variances = self.variances/10
		N = 0
		Ntot = 0
		while N < Nvalid:
			Ntot += 1
			if self._newpoint(*args):
				N += 1
				self.points.append(self.point)
				self.lh_list.append(self.lh)
		self.increasecounter(Ntot)


	def _newpoint(self, *args):
		valid = False
		while not valid:
			p0 = self.point + np.random.randn(self.Npars) * self.variances
			valid = self.inthebox(p0)
		lh0 = self.likelihood(p0, *args)
		if (lh0 > self.lh) or (lh0 > self.lh + np.log(np.random.uniform())): #Acceptance condition for log-likelihoods!!!
			self.point = p0
			self.lh = lh0
			return True
		else:
			return False


