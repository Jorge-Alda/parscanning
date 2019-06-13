from .scan import Scan
from numpy import linspace
from itertools import product

class GridScan(Scan):
	def run(self):
		Ntot = 0
		if isinstance(self.N_iters, int):
			self.N_iters = [self.N_iters,]*self.Npars
		ranges=[]
		for i in range(0, self.Npars):
			ranges.append(linspace(self.par_min[i], self.par_max[i], self.N_iters[i]))
		for r in product(*ranges):
			self.points.append(r)
			lh = self.likelihood(r)
			self.lh_list.append(lh)
			Ntot += 1
		self.increasecounter(Ntot)		 
