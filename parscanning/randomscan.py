import numpy as np
from .scan import Scan

class RandomScan(Scan):
	def run(self, *args):
		Ntot = 0
		for _ in range(0, self.N_iters):
			point = []
			for p in range(0, self.Npars):
				point.append((self.par_max[p]-self.par_min[p])*np.random.random() + self.par_min[p])
			lh = self.likelihood(point, *args)
			self.points.append(point)
			self.lh_list.append(lh)
			Ntot += 1
		self.increasecounter(Ntot)

