from .scan import Scan
from multiprocessing import Pool
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


class MLScan(Scan):
    def __init__(self, likelihood, par_min, par_max, N_iters, bf, cutoff=None):
        Scan.__init__(self, likelihood, par_min, par_max, N_iters)
        self.bf = np.array(bf)
        self.lh = self.likelihood(self.bf)
        self.cutoff = cutoff
        self.lenf = len(self.bf)
        self.featurenames = [f'f{i}' for i in range(self.lenf)]

    def init_ML(self, model, varsize=0.5):
        self.model = model
        self.started_pred = False
        self.variances = np.zeros(self.Npars)
        for p in range(0, self.Npars):
            self.variances[p] = (self.par_max[p]-self.par_min[p])*varsize

    def run(self):
        self.run_mp(1)

    def newpoints(self, num=None):
        if num is None:
            num = self.N_iters
        if 'model' not in self.__dict__.keys():
            raise NotImplementedError('You have to init the Neural Network first!')
        xlist = []
        Ntot = 0
        while len(xlist) < num:
            Ntot += 1
            x0 = self.bf + np.random.randn(self.Npars) * self.variances
            if self.guess_single(x0) > self.lh + np.log(np.random.uniform()):
                xlist.append(x0)
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

    def train_pred(self, X, y, metrics=None):
        train_X, val_X, train_y, val_y = train_test_split(X, y)
        if isinstance(X, pd.DataFrame):
            self.featurenames = list(X.keys())
        self.val_X = val_X
        self.val_y = val_y
        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)])
        self.started_pred = True
        if metrics is not None:
            val_pred = self.model.predict(val_X)
            print(metrics(val_pred, val_y))    

    def save_validation(self, fname):
        data = pd.DataFrame(self.val_X)
        data['logL'] = self.val_y
        data.to_csv(fname, sep='\t', header=False, index=False)

    def guess_lh(self, x):
        return self.model.predict(x)
        
    def guess_single(self, x):
        dfx = pd.DataFrame(np.array(x).reshape(1, self.lenf), columns=self.featurenames)
        return float(self.model.predict(dfx))

    def saveML(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.model, f)

    def loadML(self, fname):
        with open(fname, 'rb') as f:
            self.model = pickle.load(f)
            self.started_pred = True
