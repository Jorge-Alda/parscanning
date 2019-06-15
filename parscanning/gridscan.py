from .scan import Scan
import numpy as np
from itertools import product
from multiprocessing import Pool

class GridScan(Scan):
	def run(self, *args):
		Ntot = 0
		if isinstance(self.N_iters, int):
			self.N_iters = [self.N_iters,]*self.Npars
		ranges=[]
		for i in range(0, self.Npars):
			ranges.append(np.linspace(self.par_min[i], self.par_max[i], self.N_iters[i]))
		for r in product(*ranges):
			self.points.append(r)
			lh = self.likelihood(r, *args)
			self.lh_list.append(lh)
			Ntot += 1
		self.increasecounter(Ntot)

	def run_mp(self, cores, *args):		 
		if isinstance(self.N_iters, int):
			self.N_iters = [self.N_iters,]*self.Npars
		ranges=[]
		for i in range(0, self.Npars):
			ranges.append(np.linspace(self.par_min[i], self.par_max[i], self.N_iters[i]))
		points = list(product(*ranges))
		with Pool(processes=cores) as pool:
			argl = []
			for arg in args:
				argl.append([arg])
			lh = pool.starmap(self.likelihood, product(points, *argl))	
		self.increasecounter(len(points))
		self.points += points
		self.lh_list += lh

	def meshdata(self):
		ranges=[]
		for i in range(0, self.Npars):
			ranges.append(np.linspace(self.par_min[i], self.par_max[i], self.N_iters[i]))
		lhgrid = np.zeros(self.N_iters)
		rc = []
		for i in range(0, self.Npars):
			rc.append(range(self.N_iters[i]))
		for t, c in enumerate(product(*rc)):
			lhgrid[c] = self.lh_list[t]
		return (*ranges, lhgrid)

