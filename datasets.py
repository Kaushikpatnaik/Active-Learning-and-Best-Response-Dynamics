from numpy import sqrt, array, ones, zeros, dot, sign, float64, hstack, vstack, random
from scipy.stats import bernoulli, uniform
from scipy.stats import norm as normal
from scipy.linalg import norm
import cPickle
from numpy import random as rnd
from adversary import *
from matplotlib import pyplot as plt
from itertools import *

class DataSet(object):
    
    def __init__(self, d, extend = False, norm_p = None, noise = 0):
        self.d = d
        self.extend = extend
        self.norm_p = norm_p
        if not isinstance(noise, Adversary):
            self.noise = RandomClassificationNoise(float(noise))
        else:
            self.noise = noise
        
        self.oracle = dict()
        self.ex_id = 0
        self.use_last = False
        self.saved = None
    
    def initialize(self, shuffle = True):
        if shuffle == True:
            self.use_last = False
            self.saved = None
        else:
            self.use_last = True
    
    def normalize(self, p):
        self.norm_p = p
    
    def _next(self):
        raise NotImplementedError
    
    def next(self):
        x, y = self._next()
        if self.extend: 
            x = hstack((array([1.0]), x))
        if self.norm_p:
            try:
                normalized = x / norm(x, self.norm_p)
            except RuntimeWarning:
                print 'x:', x
                print '||x||_%g:' % self.norm_p, norm(x, self.norm_p)
                normalized = x / norm(x + 1e-20, self.norm_p)
            x = normalized
        return self.noise.corrupt(x, y)
    
    def labeled_examples(self, num_examples):
        unlabeled = []
        labels = []
        
        for i in range(num_examples):
            x, y = self.next()
            unlabeled.append(x)
            labels.append(y)
        
        unlabeled = array(unlabeled)
#        print unlabeled.shape, num_examples, self.d
        assert unlabeled.shape == (num_examples, self.d)
        labels = array(labels).reshape((num_examples, 1))
        
        return unlabeled, labels
    
    def unlabeled_examples(self, num_examples, oracle = False):
       
        if self.use_last and self.saved is not None:
            return self.saved
        
        unlabeled = []
        if oracle:
            ids = range(self.ex_id, self.ex_id + num_examples)
        
        for i in range(num_examples):
            x, y = self.next()
            #print x, y
            '''
            if x.all() == zeros((self.d)).all():
				print 'raising value error'
				raise ValueError
			'''
            if oracle:
                self.oracle[self.ex_id] = y
                self.ex_id += 1
            unlabeled.append(x)
        
        unlabeled = array(unlabeled)
#        print unlabeled.shape, num_examples, self.d
        assert unlabeled.shape  == (num_examples, self.d)
        
        if oracle:
            return_value = unlabeled, ids
        else:
            return_value = unlabeled
        
        if self.use_last:
            self.saved = return_value
        
        return return_value
    


class GaussianLinearSep(DataSet):
    
    def __init__(self, d, noise = 0, w = None):
        DataSet.__init__(self, d, noise = noise)
        self.dist = normal(loc = 0.0, scale = 1.0/sqrt(d))
        
        if w is None:
            self.w = self.dist.rvs(size = self.d)
	    #print self.w
        else:
            self.w = w.astype(float64)
    
    def _next(self):
        x = self.dist.rvs(size = self.d)
	#print x
        y = sign(dot(self.w, x))
        return x, y


class GaussianLinearSepMargin(DataSet):
    
    def __init__(self, d, p = 2.0, q = 2.0, margin = 0.5, noise = 0, w = None):
        DataSet.__init__(self, d, noise = noise)
        self.p = p
        self.q = q
        self.margin = margin
        self.dist = normal(loc = 0.0, scale = 1.0/sqrt(d))
        
        if w is None:
            self.w = self.dist.rvs(size = self.d)
        else:
            self.w = w.astype(float64)
        
        if self.q != None:
            self.w /= norm(self.w, self.q)
        
    
    def _next(self):
        while True:
            x = self.dist.rvs(size = self.d)
            x_margin = abs(dot(self.w, x))
            if self.p != None:
                x_margin /= norm(x, self.p)
            if x_margin >= self.margin:
                break
        y = sign(dot(self.w, x))
        return x, y


class RealDataSet(DataSet):
    
    def __init__(self, filename, shuffle, repeat, scale_p = None, data_noise =0):
        self.data = cPickle.load(open(filename, 'r'))
        self.n = len(self.data)
        self.shuffle = shuffle
        self.repeat = repeat
        self.maxnorm = self.calculatemaxnorm()
        #d = self.data[0][0].shape[0] + 1
        d = len(self.data[0][0]) + 1
        DataSet.__init__(self, d, extend = True, norm_p = False, noise = data_noise)
        
        if scale_p:
            max_norm = 0.0
            for x, y in iter(self.data):
                cur_norm = norm(x, scale_p)
                if cur_norm > max_norm:
                    max_norm = cur_norm
            print max_norm
            for i in range(self.n):
                x, y = self.data[i]
                self.data[i] = (1000 * x / max_norm, y)
    
    def __str__(self):
        return self.__class__.__name__
    
    def initialize(self, shuffle = True):
        if self.shuffle and shuffle:
            random.shuffle(self.data)
        self.iterator = iter(self.data)
        #print len(self.data)
        
    def iterate(self):
		self.iterator = iter(self.data)
		
    def calculatemaxnorm(self):
        maxnorm = 0
        for i in range(self.n):
            p, q = self.data[i]
            if norm(p,2) > maxnorm:
                maxnorm = norm(p, 2.0)
            else:
                continue
        #print maxnorm
        return maxnorm
    
    def _next(self):
        try:
			x, y = self.iterator.next()
        except StopIteration:
            if self.repeat:
                self.iterator = iter(self.data)
                x, y = self.iterator.next()
            else:
				print 'Exceeded sampling of dataset, breaking loop'
				raise StopIteration
        #print x, y
        return x, y


class TestDataSet(DataSet):
    
    def __init__(self, d, dist1mean, dist2mean, cov, size, noise = 0, w = None):
        
        x = rnd.multivariate_normal(dist1mean, cov, size)
        #z = rnd.multivariate_normal(dist2mean, cov, size)
        
        #z = zeros((size,2))
        #x = zeros((size,2))
        
        '''
        i=0
        while i < size:
            temp = rnd.multivariate_normal(dist1mean,cov)
            if norm(temp,2) <= 1:
				z[i] = temp
				i += 1

        i=0
        while i < size:	
            temp = rnd.multivariate_normal(dist1mean,cov)
            if norm(temp,2) > 1.2:
				x[i] = temp
				i += 1
        
        P = []
        Q = []
        R = []
        S = []
        
        for i in range(size):
            p,q = x[i]
            P.append(p)
            Q.append(q)
            r,s = z[i]
            R.append(r)
            S.append(s)

        #plt.clf()

        #plt.plot(P,Q,'bo')
        #plt.plot(R,S,'g+')
        #plt.show()
        '''
        temp = []

        for i in range(size):
			# change here for separable and non separable dataset
            #temp.append([z[i],1])
            #temp.append([x[i],-1])
            temp.append(z[i])
            #temp.append(x[i])

        self.data = temp
        self.n = size
        self.shuffle = True
        self.repeat = True
        self.maxnorm = self.calculatemaxnorm()
        DataSet.__init__(self, d, extend = False, norm_p = True, noise = noise)

        if w is None:
            self.w = normal(loc = 0.0, scale = 1.0/sqrt(d))
        else:
            self.w = w.astype(float64)
    
    def iterate(self):
		self.iterator = iter(self.data)
	
    def initialize(self, shuffle = True):
        if self.shuffle and shuffle:
            random.shuffle(self.data)
        self.iterator = iter(self.data)
        
    def calculatemaxnorm(self):
        maxnorm = 0
        for i in range(self.n):
			# change here for separable and non-separable dataset
            #p, q = self.data[i]
            p = self.data[i]
            if norm(p,2) > maxnorm:
                maxnorm = norm(p, 2.0)
            else:
                continue
        print maxnorm
        return maxnorm
    
    def _next(self):
        try:
			# change here for separable and non separable dataset
			#x, y = self.iterator.next()
			x = self.iterator.next()
			y = sign(dot(self.w, x))
        except StopIteration:
            if self.repeat:
				# change here for separable and non separable dataset
                self.iterator = iter(self.data)
                #x, y = self.iterator.next()
                x = self.iterator.next()
                y = sign(dot(self.w, x))
            else:
				print 'Exceeded sampling of dataset, breaking loop'
				raise StopIteration
        return x, y
        
    def optimal_accuracy(self, dist1mean, dist2mean):
		midpoint = (dist1mean - dist2mean)
		weight_vector = midpoint
		weight_vector /= norm(weight_vector,2)
		count_correct = 0.0
		
		for i in range(self.n):
			x, y = self.data[i]
			if sign(dot(weight_vector,x)) == sign(y):
				count_correct += 1
		
		accuracy = count_correct/self.n;
		return accuracy

class UniformTestDataSet(DataSet):

    def __init__(self, d, data, labels, size):
        
        self.data = data
        self.labels = labels
        self.n = size
        self.dist = []
        for i in izip(data, labels):
            self.dist.append(i)

        self.shuffle = True
        self.repeat = True
        DataSet.__init__(self, d, extend = False, norm_p = False, noise = 0.0)

    
    def initialize(self, shuffle = True):

        if self.shuffle and shuffle:
            random.shuffle(self.dist)
        self.iterator = iter(self.dist)
            
    def _next(self):
        try:
            x, y = self.iterator.next()
        except StopIteration:
            if self.repeat:
                self.iterator = iter(self.dist)
                x, y = self.iterator.next()
            else:
                print 'Exceeded sampling of dataset, breaking loop'
                raise StopIteration
        return x, y






