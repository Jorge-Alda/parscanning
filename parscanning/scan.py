import time
from numpy.linalg import norm
import numpy as np

class Scan:
	def __init__(self, likelihood, par_min, par_max, N_iters):
		self.likelihood = likelihood
		if isinstance(par_min, float) and isinstance(par_max, float):
			self.Npars = 1
			self.par_min = [par_min,]
			self.par_max = [par_max,]
		elif len(par_min) == len(par_max):
			self.Npars = len(par_min)
			self.par_min = par_min
			self.par_max = par_max
		else:
			raise Exception("The length of the bounds of the parameter doesn't match!")
		self.N_iters = N_iters
		self.points = []
		self.lh_list = []
		self.Ntot = 0 
	
	def run(self):
		raise NotImplementedError("You have to define the scan")
	
	def run_time(self):
		self.start = time.time()
		self.run()
		self.end = time.time()
		print("Running time: " + str(self.end-self.start) + 's')

	def run_mp(self, cores):
		from multiprocessing import Process, Manager
		with Manager() as manager:
			self.points = manager.list(self.points)
			self.lh_list = manager.list(self.lh_list)
			processes = []
			for i in range(0, cores):
				p = Process(target = self.run)
				p.start()
				processes.append(p)
			for p in processes:
				p.join()
			self.points = list(self.points)
			self.lh_list = list(self.lh_list) 
	
	def clear(self):
		self.points = []
		self.lh_list = []

	def get_points(self):
		return self.points

	def get_lh_list(self):
		return self.lh_list

	def get_point_series(self, coord):
		s = []
		for i in range(0, len(self.points)):
			s.append(self.points[i][coord])
		return s

	def interpolate(self, point):
		num = 0
		den = 0
		for i in range(0, len(self.points)):
			d = norm(np.array(point) - np.array(self.points[i]))
			if d == 0:
				return self.lh_list[i]
			else:
				num += self.lh_list[i]/d**4
				den += 1/d**4
		return num/den

	def write(self, fout):
		f = open(fout, 'at')
		for p, l in zip(self.points, self.lh_list):
			for i in range(0, self.Npars):
				f.write(str(p[i])+'\t')
			f.write(str(l)+'\n')
		f.close()

	def inthebox(self, point):
		for p in range(0, self.Npars):
			if point[p] < self.par_min[p]:
				return False
			if point[p] > self.par_max[p]:
				return False
		return True

	def acceptance(self):
		return len(self.points)/self.Ntot
