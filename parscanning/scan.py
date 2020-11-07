import time
from numpy.linalg import norm
import numpy as np
from multiprocessing import Process, Manager, Lock, Pool
from ctypes import c_int
from itertools import product

Ntotlock = Lock()

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
			raise Exception("The length of the limits of the parameter doesn't match!")
		self.N_iters = N_iters
		self.points = []
		self.lh_list = []
		self.Ntot = 0
		self.mp = False 
	
	def run(self, *args):
		raise NotImplementedError("You have to define the scan")
	
	def run_time(self, *args):
		self.start = time.time()
		self.run(*args)
		self.end = time.time()
		print("Running time: " + str(self.end-self.start) + 's')

	def increasecounter(self, Ntot):
		if self.mp:
			with Ntotlock:
				self.Ntot.value += Ntot
		else:
			self.Ntot += Ntot

	def run_mp(self, cores, *args):
		self.mp = True
		with Manager() as manager:
			self.points = manager.list(self.points)
			self.lh_list = manager.list(self.lh_list)
			self.Ntot = manager.Value(c_int, self.Ntot)
			processes = []
			for i in range(0, cores):
				p = Process(target = self.run, args = args)
				p.start()
				np.random.seed(int(p.pid + time.time())) #We have to reseed each process, or else they will produce the same random numbers
				processes.append(p)
			for p in processes:
				p.join()
			self.points = list(self.points)
			self.lh_list = list(self.lh_list)
			self.Ntot = int(self.Ntot.value)
		self.mp = False 

	def run_mp_time(self, cores):
		self.start = time.time()
		self.run_mp(cores)
		self.end = time.time()
		print("Running time: " + str(self.end-self.start) + 's')
	
	def clear(self):
		self.points = []
		self.lh_list = []
		self.Ntot = 0

	def get_points(self):
		return self.points

	def get_lh_list(self, index=None):
		if index == None:
			return self.lh_list
		else:
			s = []
			for i in range(0, len(self.lh_list)):
				s.append(self.lh_list[i][index])
			return s

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

	def write(self, fout, mode='wt'):
		with open(fout, mode) as f:
			for p, l in zip(self.points, self.lh_list):
				for i in range(0, self.Npars):
					f.write(str(p[i])+'\t')
				f.write(str(l)+'\n')

	def inthebox(self, point):
		for p in range(0, self.Npars):
			if point[p] < self.par_min[p]:
				return False
			if point[p] > self.par_max[p]:
				return False
		return True

	def acceptance(self):
		return len(self.points)/self.Ntot

	def bestpoint(self):
		return self.points[np.argmax(self.lh_list)]

	def expectedvalue(self, func, *args):
		lhmax = np.max(self.lh_list)
		num = 0
		den = 0
		for p, l in zip(self.points, self.lh_list):
			expl = np.exp(l-lhmax)
			num += func(p, *args) * expl
			den += expl
		return num/den

	def expectedvalue_mp(self, func, cores, *args):
		lhmax = np.max(self.lh_list)
		num = 0
		den = 0
		with Pool(processes=cores) as pool:
			argl = []
			for arg in args:
				argl.append([arg])
			flist = pool.starmap(func, product(self.points, *argl))
		for i, l in enumerate(self.lh_list):
			expl = np.exp(l-lhmax)
			num += flist[i] * expl
			den += expl
		return num/den		
