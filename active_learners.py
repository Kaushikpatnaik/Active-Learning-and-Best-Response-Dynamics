from numpy import *
import numpy.random as nprandom
from scipy.stats import bernoulli
from scipy.stats import norm as normal
from scipy import linalg as lin
import math
import random as rnd

from learners import *
from passive_learners import *
from utilities import *
import gc

#SOLVER = 'sgd'
SOLVER = 'cvxopt'
#SOLVER = 'hard_cvxopt'
#SOLVER = 'svm'

class PassiveSVM(LinearLearner, ActiveBatchLearner, ActiveSourceLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d)
        self.C = C
     
    def active_batch_train(self, U, ids, oracle, label_budget):
        m = min(len(U), label_budget)
        
        # Query the labels of m examples
        Y = array([oracle(ids[i]) for i in range(m)]).reshape((m, 1))
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        svm.batch_train(U[:m], Y)
        
        # Use the separator found by SVM
        self.w = svm.w
    
    def active_source_train(self, source, oracle, label_budget):
        U, ids = source(label_budget)
        self.active_batch_train(U, ids, oracle, label_budget)


class SimpleMarginBatch(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d):
        LinearLearner.__init__(self, d, w = None)
        self.pointsusage = 0
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        m = len(U)
        label_budget = min(m, label_budget)
        self.pointsusage = label_budget
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        
        # Query the labels of the first 10 examples
        start = min(10, label_budget)
        for i in range(label_budget):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
        
        # Run standard SVM on the labeled data
        svm = SVM(self.d)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find the unlabeled example with the smallest margin
            min_margin = 1000
            min_index = 0
            for i in range(m-10):
                if ids[i] in used:
                    continue
                cur_margin = svm.margin(U[i])
                if cur_margin < min_margin:
                    min_margin = cur_margin
                    min_index = i
            
            # Query its label
            X = vstack((X, U[min_index]))
            Y = vstack((Y, array([oracle(ids[min_index])]).reshape((1,1))))
            used.add(ids[min_index])
            
            # Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w


class SimpleMarginSource(LinearLearner, ActiveSourceLearner):
    
    def __init__(self, d):
        LinearLearner.__init__(self, d, w = None)
    
    def active_source_train(self, source, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        
        # Query the labels of 10 examples
        start = min(10, label_budget)
        U, ids = source(start)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
        
        # Run standard SVM on the labeled data
        svm = SVM(self.d)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find an unlabeled example with a small margin
            min_margin = min(abs(svm.margin(X[i])) for i in range(len(used)))
            U, ids = source(1)
            while abs(svm.margin(U[0])) >= min_margin:
                U, ids = source(1)
            
            # Query its label
            X = vstack((X, U[0]))
            Y = vstack((Y, array([oracle(ids[0])]).reshape((1,1))))
            used.add(ids[0])
            
            # Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w

class LinearSeparatorsNoiseSource(LinearLearner, ActiveSourceLearner):
	def __init__(self, d, eps, c1, c2, c3, c4):
		LinearLearner.__init__(self, d, w = None)
		self.eps = eps
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3
		self.c4 = c4
	
	def active_source_train(self, source, oracle, label_budget):
	
		itr = int(log(1/self.eps)/log(2)) - 2
		m = zeros((itr+1))
		b = zeros((itr+1))
		r = zeros((itr+1))
		n = zeros((itr+1))
		t = zeros((itr+1))
		#print itr
		
		# Generator function for mk, bk, rk, nk, tk
		for i in range(0,itr+1):
			b[i] = self.c1*pow(2,-i) 				# * pow(self.d,-1/2.0)) 	# cutoff values
			r[i] = self.c2*(pow(2,-i) * pi) 			# radius cutoff
			#n[i] = label_budget/pow(2,itr-i-1)		# train is the number of unlabelled examples allowed, this limits the unlabelled samples per iteration of svm <needs to be increase exponentially >
			#m[i] = self.c3*(pow(d,2) + d*i)/3			# maxm number of labels allowed
			m[i] = self.c3*label_budget/itr
			t[i] = self.c4*pow(2,-i) 				# * pow(self.d,-1/2.0))
		
		#print "reached Generator function in Linear Noise Separators"
		
		# access to the training examples and their labels		
		X = array([]).reshape((0, self.d))
		Y = array([]).reshape((0,1))
		c = set()
		
		U, ids = source(int(m[0]))
		for i in range(int(m[0])):
			X = vstack((X, U[i]))
			Y = vstack((Y, array([oracle(ids[i])]).reshape(1,1)))
			c.add(ids[i])
		
		#print "sending data to the PCA"
		
		# PCA code for finding the finding the eigenvectors and eigenvalues. <limit data by exp(dim/8), not implemented, does not need to be>
		pca = PCA(self.d)
		pca.pca_run(X,Y)
		self.w = pca.w
		
		# Creating the working set and the temp arrays for the weight update
		workingset = []
		tempx = array([]).reshape((0,self.d))
		k = 1
		
		# populating workingset with m[1]
		U, ids = source(int(m[1]))
		for i in range(int(m[1])):
			workingset.append(ids[i])
			tempx = vstack((tempx, U[i]))
		
		while k < itr:
		
			#print "iteration number: "+str(k)
			
			# Creating set for SVM
			P = array([]).reshape((0, self.d))
			Q = array([]).reshape((0, 1))
			
			# Querying labels, subsampling points from workingset, to get labels for optimization problem
			for i in range(int(m[k])):
				P = vstack((P, tempx[i]))
				Q = vstack((Q, array([oracle(workingset[i])]).reshape(1,1)))
			
			# QP to find w_improved, also need to send rk, tk, wk-1
			qp = QP(self.d)
			qp.train(P, Q, r[k], t[k], self.w)
			self.w = qp.w
			
			# clearing the working set and making tempx zero
			workingset = []
			tempx = array([]).reshape((0,self.d))
			
			#print 'query additional points from the oracle returning x based on != |w_improved.x| >= bk'
			while len(workingset) <= int(m[k+1]):
				U, ids = source(1)
				if abs(dot(self.w, U[0])) >= b[k]:
					continue
				else:
					workingset.append(ids[0])
					tempx = vstack((tempx, U[0]))
			k +=1

class SimpleMarginSoftSVMSource(LinearLearner, ActiveSourceLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
    
    def active_source_train(self, source, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        
        # Query the labels of 10 examples
        start = min(10, label_budget)
        U, ids = source(start)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find an unlabeled example with a small margin
			min_margin = min(abs(svm.margin(X[i])) for i in range(len(used)))
			
			U, ids = source(1)
			while abs(svm.margin(U[0])) >= min_margin:
				U, ids = source(1)
			
			# Query its label
			X = vstack((X, U[0]))
			Y = vstack((Y, array([oracle(ids[0])]).reshape((1,1))))
			used.add(ids[0])
			
			# Run SVM on all the labeled data
			svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w


class AverageMarginSoftSVMSource(LinearLearner, ActiveSourceLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
    
    def active_source_train(self, source, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        
        # Query the labels of 10 examples
        start = min(10, label_budget)
        U, ids = source(start)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find an unlabeled example with a small margin
            avg_margin = sum(Y[i] * svm.margin(X[i]) for i in range(len(used)))
            
            U, ids = source(1)
            while abs(svm.margin(U[0])) >= 0.5 * avg_margin:
				U, ids = source(1)
            
            # Query its label
            X = vstack((X, U[0]))
            Y = vstack((Y, array([oracle(ids[0])]).reshape((1,1))))
            used.add(ids[0])
            
            # Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w

class LinearNoiseMethod2Source(LinearLearner, ActiveSourceLearner):
	def __init__(self, d, eps, c1, c2, c3, c4):
		LinearLearner.__init__(self, d, w = None)
		self.eps = eps
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3
		self.c4 = c4
	
	def active_source_train(self, source, oracle, label_budget):
	
		itr = int(log(1/self.eps)/log(2))+2
		#print "total iterations:"+str(itr)
		m = zeros((itr+1))
		b = zeros((itr+1))
		r = zeros((itr+1))
		n = zeros((itr+1))
		t = zeros((itr+1))
		
		#print itr
		
		# Generator function for mk
		for i in range(0,itr+1):
			m[i] = label_budget/itr
		
		# Estimating parameters
		
		# access to the training examples and their labels		
		X = array([]).reshape((0, self.d))
		Y = array([]).reshape((0,1))
		c = set()
		
		U, ids = source(int(m[0]))
		for i in range(int(m[0])):
			X = vstack((X, U[i]))
			Y = vstack((Y, array([oracle(ids[i])]).reshape(1,1)))
			c.add(ids[i])
		
		# learning a more accurate weight vector using the svm algorithm (can be replaced by hinge loss minimization later)
		svm = SVM(self.d)
		svm.batch_train(X, Y)
		
		tot = 0
		avgtot = 0
		# Calculating the normalized band of svm
		for i in range(int(m[0])):
			avgtot += abs(dot(svm.w,X[i])) * 1/(abs(dot(svm.w, X[i])))
			tot += 1/abs(dot(svm.w,X[i]))
		
		b = avgtot/tot
		r = b * pow(self.d, 1/2) * pi
		t = b
		
		#print "cutoff, normalizing factor, radius"
		#print b,t,r
		
		# access to the training examples and their labels		
		tempx = array([]).reshape((0, self.d))
		workingset = []
		
		U, ids = source(int(m[1]))
		for i in range(int(m[1])):
			tempx = vstack((tempx, U[i]))
			workingset.append(ids[i])
		
		k=1
		
		while (k < itr):
		
			#print "iteration number: "+str(k)
			
			# Creating set for SVM
			P = array([]).reshape((0, self.d))
			Q = array([]).reshape((0, 1))
			
			# Querying labels, subsampling points from workingset, to get labels for optimization problem
			for i in range(int(m[k])):
				P = vstack((P, tempx[i]))
				Q = vstack((Q, array([oracle(workingset[i])]).reshape(1,1)))
			
			# QP to find w_improved, also need to send rk, tk, wk-1
			qp = QP(self.d)
			qp.train(P, Q, r, t, self.w)
			self.w = qp.w
			
			tot = 0
			avgtot = 0
			# Calculating the normalized band of svm
			for i in range(int(m[k])):
				avgtot += abs(dot(self.w,P[i])) * 1/(abs(dot(self.w, P[i])))
				tot += 1/abs(dot(self.w,P[i]))
			
			b = avgtot/tot
			r = b * pow(self.d, 1/2) * pi
			t = b
			
			# clearing the working set and making tempx zero
			workingset = []
			tempx = array([]).reshape((0,self.d))
				
			#print "cutoff, normalizing factor, radius"
			#print b,t,r
			
			# query additional points from the oracle returning x based on != |w_improved.x| >= b
			while len(workingset) <= int(m[k+1]):
				U, ids = source(1)
				if abs(dot(self.w, U[0])) >= b:
					continue
				else:
					workingset.append(ids[0])
					tempx = vstack((tempx, U[0]))
			
			k +=1

class LinearSeparatorsNoiseBatch(LinearLearner, ActiveBatchLearner):
	def __init__(self, d, eps, c1, c2, c3, c4):
		LinearLearner.__init__(self, d, w = None)
		self.eps = eps
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3
		self.c4 = c4
	
	def active_batch_train(self, U, ids, oracle, label_budget):
	
		itr = int(log(1/self.eps)/log(2)) - 2
		m = zeros((itr+1))
		b = zeros((itr+1))
		r = zeros((itr+1))
		n = zeros((itr+1))
		t = zeros((itr+1))
		count = len(U)   							# Here U is set to a fraction of the dataset

		# Generator function for mk, bk, rk, nk, tk
		for i in range(0,itr+1):
			b[i] = self.c1*pow(2,-i) 				# * pow(self.d,-1/2.0)) 	# cutoff values
			r[i] = self.c2*(pow(2,-i) * pi) 			# radius cutoff
			m[i] = int(self.c3*label_budget/itr)
			t[i] = self.c4*pow(2,-i) 				# * pow(self.d,-1/2.0))
		
		# access to the training examples and their labels		
		X = array([]).reshape((0, self.d))
		Y = array([]).reshape((0,1))
		used = set()
		
		for i in range(int(m[0])):
			X = vstack((X, U[i]))
			Y = vstack((Y, array([oracle(ids[i])]).reshape(1,1)))
			used.add(ids[i])
		
		# PCA code for finding the finding the eigenvectors and eigenvalues. <limit data by exp(dim/8), not implemented, does not need to be>
		pca = PCA(self.d)
		pca.pca_run(X,Y)
		self.w = pca.w

		# Creating the working set and the temp arrays for the weight update
		workingset = []
		tempx = array([]).reshape((0,self.d))
		k = 1
		
		use_count = len(used)
		# populating workingset with m[1]
		for i in range(int(m[1])):
			workingset.append(ids[use_count+i])
			tempx = vstack((tempx, U[use_count+i]))
			used.add(ids[use_count+i])
		
		while (k < itr) and (len(used) < len(U)):
		
			#print 'iteration number, margin, norm, radius :'
			#print str(k),str(b[k]),str(t[k]),str(r[k])
			#print " datapoints used, datapints sampled"
			#print len(workingset), len(used)
			
			# Creating set for SVM
			P = array([]).reshape((0, self.d))
			Q = array([]).reshape((0, 1))
			
			# Querying labels, subsampling points from workingset, to get labels for optimization problem
			for i in range(int(m[k])):
				P = vstack((P, tempx[i]))
				Q = vstack((Q, array([oracle(workingset[i])]).reshape(1,1)))
				
			
			# QP to find w_improved, also need to send rk, tk, wk-1
			qp = QP(self.d)
			qp.train(P, Q, r[k], t[k], self.w)
			self.w = qp.w

			'''
			# You will operate in k-1 space, for k=2 you are looking at the difference between k=1 self.w and k=2 self.w
			if k > 1:
				dis_disregion =0.0
				dis_aggregion =0.0
				ratio =0.0
				for i in range(len(U)):
					if sign(dot(old_w,U[i])) != sign(dot(self.w,U[i])):
						if abs(dot(old_w, U[i])) < b[k-1]:
							dis_disregion += 1
						else:
							dis_aggregion += 1
				print 'In iteration ' + str(k-1) + ' disagreement within margin is ' + str(dis_disregion) + ' disagreement outside margin is ' + str(dis_aggregion)
				if dis_disregion ==0:
					ratio=0
				else:
					ratio= float(dis_aggregion/dis_disregion)
				print 'Ratio of disagreement is ' + str(ratio)
				print '\n'
			'''
			old_w = self.w
			
			# clearing the working set and making tempx zero
			workingset = []
			tempx = array([]).reshape((0,self.d))
			
			#print self.w
			while (len(workingset) <= int(m[k+1])) and (len(used) < count):
				p = len(used)
				used.add(ids[p])
				if abs(dot(self.w, U[p])) >= b[k]:
					continue
				else:
					workingset.append(ids[p])
					tempx = vstack((tempx, U[p]))
			k +=1

class SimpleMarginSoftSVMBatch(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
        self.initial_sample = 5
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        labeled = []
        
        # Query the labels of some examples
        start = min(self.initial_sample, label_budget)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
            labeled.append(ids[i])
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find the unlabeled example with the smallest margin
            margins = abs(svm.margin(U))
            min_margin = inf
            min_index = 0
            for i in range(len(U)):
                if ids[i] in used:
                    continue
                cur_margin = margins[i]
                if cur_margin < min_margin:
                    min_margin = cur_margin
                    min_index = i
            
            # Query its label
            X = vstack((X, U[min_index]))
            Y = vstack((Y, array([oracle(ids[min_index])]).reshape((1,1))))
            used.add(ids[min_index])
			
			# Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w


class MaxMinMarginSoftSVMBatch(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
        self.initial_sample = 5
        self.pointsusage = 0
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        labeled = []
        self.pointsusage = label_budget
        
        # Query the labels of some examples
        start = min(self.initial_sample, label_budget)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
            labeled.append(ids[i])
        
        # Create standard SVM object
        svm = soft_SVM(self.d, self.C)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find the unlabeled example with the largest min-margin
            max_min_margin = 0
            max_index = 0
            for i in range(len(U)):
                if ids[i] in used:
                    continue
                
                # Compute the two margins and take the min
                cur_margins = []
                for label in (-1, 1):
                    new_X = vstack((X, U[i]))
                    new_Y = vstack((Y, array([label]).reshape((1,1))))
                    svm.batch_train(new_X, new_Y)
                    cur_margin = min(abs(svm.margin(new_X[i])) for i in range(len(new_X)))
                    cur_margins.append(cur_margin)
                
                cur_min_margin = min(cur_margins)
                
                if cur_min_margin > max_min_margin:
                    max_min_margin = cur_min_margin
                    max_index = i
            
            # Query its label
            X = vstack((X, U[max_index]))
            Y = vstack((Y, array([oracle(ids[max_index])]).reshape((1,1))))
            used.add(ids[max_index])
			
		# Run SVM on all the labeled data
        svm.batch_train(X, Y)
        self.w = svm.w


class RatioMarginSoftSVMBatch(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
        self.initial_sample = 5
        self.pointsusage = 0
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        labeled = []
        self.pointsusage = label_budget
        
        # Query the labels of some examples
        start = min(self.initial_sample, label_budget)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
            labeled.append(ids[i])
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            # Find the unlabeled example with the largest min-margin-ratio
            max_min_margin_ratio = 0
            max_index = 0
            for i in range(len(U)):
                if ids[i] in used:
                    continue
                
                # Compute the two margins and take the min
                cur_margins = []
                for label in (-1, 1):
                    new_X = vstack((X, U[i]))
                    new_Y = vstack((Y, array([label]).reshape((1,1))))
                    svm.batch_train(new_X, new_Y)
                    cur_margin = min(abs(svm.margin(new_X[i])) for i in range(len(new_X)))
                    cur_margins.append(cur_margin)
                
                margin_ratios = (cur_margins[0] / cur_margins[1],
                                 cur_margins[1] / cur_margins[0])
                cur_min_margin_ratio = min(margin_ratios)
                
                if cur_min_margin_ratio > max_min_margin_ratio:
                    max_min_margin_ratio = cur_min_margin_ratio
                    max_index = i
            
            # Query its label
            X = vstack((X, U[max_index]))
            Y = vstack((Y, array([oracle(ids[max_index])]).reshape((1,1))))
            used.add(ids[max_index])
			
        # Run SVM on all the labeled data
        svm.batch_train(X, Y)
        self.w = svm.w


class AverageMarginSoftSVMBatch(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        labeled = []
        
        # Query the labels of 10 examples
        start = min(10, label_budget)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
            labeled.append(ids[i])
        
        # Run standard SVM on the labeled data
        svm = soft_SVM(self.d, self.C)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while (len(labeled) < label_budget) and (len(used) < len(U)):
            
            # Find an unlabeled example with a small margin
            avg_margin = sum(Y[i] * svm.margin(X[i]) for i in range(len(labeled)))
            
            p = len(used)
            while abs(svm.margin(U[p])) >= 0.5 * avg_margin:
				used.add(ids[p])
				p += 1
				if p == len(U):
					break
				
            if len(used) == len(U):
				break
            
            # Query its label
            X = vstack((X, U[p]))
            Y = vstack((Y, array([oracle(ids[p])]).reshape((1,1))))
            labeled.append(ids[p])
            used.add(ids[p])
            
            # Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.w = svm.w


class MarginBasedActiveLearnerBase(LinearLearner, ActiveBatchLearner):
    
	def __init__(self, d, num_iters, constant1, constant2, constant3):
		LinearLearner.__init__(self, d, w = None)
		self.num_iters = num_iters
		self.separators = []
		self.pointsusage = 0
		self.combine_final = False
		self.constant1 = constant1
		self.constant2 = constant2
		self.constant3 = constant3
	
	def initialize_weights(self, X, Y):
		# Use PCA
		#pca = PCA(self.d)
		#pca.pca_run(X)
		#self.w = pca.w
		
		# Use averaging algo
		avg = Average(self.d)
		avg.batch_train(X,Y)
		self.w = avg.w
	
	def hinge_loss_minimization(self, X, Y, tau, w, r):

		if SOLVER == "cvxopt":
			qp = QP(self.d)
			qp.train(X, Y, r, tau, w)
			self.w = qp.w
		
		elif SOLVER == 'sgd':
			sgd = HingeLossSGD2(self.d, tau, w, r)
			sgd.batch_train(X, Y)
			self.w = sgd.w
		
		elif SOLVER == 'hard_cvxopt':
			qp = QP_hardmargin(self.d)
			qp.train(X, Y, r, tau, w)
			self.w = qp.w
		
		elif SOLVER == 'svm':
			qp = soft_SVM(self.d, tau)
			qp.batch_train(X,Y)
			self.w = qp.w
		
		else:
			raise ValueError, 'Solver %s not implemented.' % SOLVER
	
	def active_batch_train(self, U, ids, oracle, label_budget):
		
		itr = self.num_iters
		count = len(U)
		
		# Set up number of examples in each iteration
		m, n = self.compute_sizes(count, label_budget, itr)
		
		# Print some initialization info
		
		#print
		#print self.__class__.__name__
		#print 'Iterations:', itr
		#print 'm:', m
		#print 'n:', n
		
		
		# access to the training examples and their labels		
		X = array([]).reshape((0, self.d))
		Y = array([]).reshape((0,1))
		used = set()
		
		for i in range(m[0]):
			X = vstack((X, U[i]))
			Y = vstack((Y, array([oracle(ids[i])]).reshape(1,1)))
			used.add(ids[i])
		
                #print "Ids picked up in the first iteration of the active algorithm"
                #print used

		# Pick starting weight vector based on initial sample
		self.initialize_weights(X, Y)
		self.separators.append(self.w)
		
		# Keep track of all labeled data
		R = array([]).reshape((0, self.d))
		S = array([]).reshape((0, 1))
		
		for k in range(1, itr):
		
			b, t, r = self.set_parameters(U, k)

			# Print current iteration info
			#print
			#print 'k:', k
			#print 'b:', b
			#print 't:', t
			#print 'r:', r
			
			# Set of points to send to outlier removal
			P = array([]).reshape((0,self.d))
			workingset = []
			
			# Find n[k] additional points within the band
			while (len(workingset) < n[k]) and (len(used) < count):
				next = len(used)
				used.add(ids[next])
				if abs(dot(self.w, U[next])) <= b:
					P = vstack((P, U[next]))
					workingset.append(ids[next])

                        #print "Ids choosen in iteration " + str(k)
                        #print workingset  			

			# Check unlabeled data usage and break if necessary
			if len(workingset) != n[k]:
				#print
				#print 'Out of unlabled data.'
				break
			
			# Perform outlier removal on P
			chosen = self.outlier_removal(P, m[k], b, r, t, self.w)
			assert len(chosen) == m[k]
			
			# Query labels of the selected points
			X = P[chosen]
			Y = array([]).reshape((0, 1))
			for i in chosen:
				Y = vstack((Y, array([oracle(workingset[i])]).reshape(1,1)))
			
			R = vstack((R, X))
			S = vstack((S, Y))
			
			# Perform constrained hinge loss minimization on X, Y
			SOLVER = 'soft_cvxopt'
			self.hinge_loss_minimization(X, Y, t, self.w, r)
			self.separators.append(self.w)

                        #print "weight vector in iteration " + str(k)
                        #print self.w			
			
			# Print some end of iteration info?
			
		
		# Calculate the final weight vector based on points labeled so far.
		if self.combine_final:
			for i in range(len(workingset)):
				if len(S) + m[0] < label_budget:
					R = vstack((R, P[i]))
					S = vstack((S, array([oracle(workingset[i])]).reshape(1,1)))
				else:
					break
		
			self.hinge_loss_minimization(R, S, t, self.w, r)
			self.separators.append(self.w)
		
		#print "Final Weight Vector"
		#print self.w
		self.pointsusage = len(S) + m[0]
        '''
		# Print final info
		print
		print 'Label usage:'
		print 'budget:', label_budget
		print 'used:  ', self.pointsusage
		print
		print 'Unlabeled usage:'
		print 'total:', count
		print 'used: ', len(used)
        '''

class MarginBasedBasic(MarginBasedActiveLearnerBase):
	def __init__(self, d, num_iters, constant1, constant2, constant3):
		MarginBasedActiveLearnerBase.__init__(self, d, num_iters, constant1, constant2, constant3)
	
	def compute_sizes(self, unlabeled, labeled, num_iters):
		m = [0] * num_iters
		n = [0] * num_iters
		
		# Increase m_i logarithmically
		denom = sum(log(i + 2) for i in range(num_iters))
		for i in range(num_iters):
			m[i] = int(labeled * log(i + 2) / denom)

		# Add remaining label budget to last iteration
		m[num_iters - 1] += labeled - sum(m)
		
		# Default is no outlier removal, so n = m
		for i in range(num_iters):
			n[i] = m[i]
		
		return m, n

	def outlier_removal(self, P, m, b, r, tau, w):
		# No outlier removal. Return all indices.
		return array(range(m))


class MarginBasedOutlierRemoval(MarginBasedActiveLearnerBase):
	def __init__(self, d, num_iters):
		MarginBasedActiveLearnerBase.__init__(self, d, num_iters)
	
	def compute_sizes(self, unlabeled, labeled, num_iters):
		m = [0] * num_iters
		n = [0] * num_iters
		
		# Increase m_i logarithmically
		denom = sum(log(i + 2) for i in range(num_iters))
		for i in range(num_iters):
			m[i] = int(labeled * log(i + 2) / denom)

		# Add remaining label budget to last iteration
		m[num_iters - 1] += labeled - sum(m)
		
		# Number of unlabeled examples available for outlier removal
		#n_total = 0.1 * unlabeled
		n_total = 10 * labeled
		
		# Increase n_i exponentially
		denom = sum(2.0**i for i in range(0, num_iters))
		for i in range(num_iters):
			n[i] = int(n_total * 2.0**i / denom) + m[i]
		
		return m, n

	def outlier_removal(self, P, m, b, r, tau, w):
		n = P.shape[0]

		# Compute distribution Q over examples in P
		outlier = OutlierRemoval(self.d)
		outlier.train(P, b, r, tau, w, 0.1)
		Q = outlier.weightdist
		Q /= lin.norm(Q, 1)
		
		# Sample random indices based on resulting Q
		chosen = nprandom.choice(n, size = m, replace = False, p = Q)
		return chosen
	

class Theoretical:
	
	def set_parameters(self, U, k):
		# Constants for theoretical parameters
		c1 = 1.0
		c2 = 1.0
		c3 = pi
		
		# Parameters for iteration k
		b = c1 * pow(2, -k) * pow(self.d, -0.5)
		t = c2 * b
		r = c3 * pow(2, -k)
		return b, t, r


class MarginBasedTheoreticalParams(MarginBasedBasic, Theoretical):
	def __init__(self, d, num_iters, constant1, constant2, constant3):
		MarginBasedBasic.__init__(self, d, num_iters, constant1, constant2, constant3)


class MarginBasedTheoreticalParamsOR(MarginBasedOutlierRemoval, Theoretical):
	def __init__(self, d, num_iters, constant1, constant2, constant3):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters, constant1, constant2, constant3)


class VarianceMethod:
	
	def set_parameters(self, U, k):
		# Constants for the variance method
		c1 = 1.0
		c2 = 0.25
		c3 = 1.0
		
		# Compute distances and standard deviation
		dotdistance = abs(self.margin(U))
		std_dev = std(dotdistance)
		
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin")
		
		# Parameters for iteration k
		b = c1 * std_dev * pow(2,-k+2)
		t = c2 * b
		r = param.radiusparam
		return b, t, r


class LinearNoiseMethodVarianceBatch(MarginBasedBasic, VarianceMethod):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodVarianceBatchOR(MarginBasedOutlierRemoval, VarianceMethod):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class AllExp:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "exp", label)
	
		#print "parameters selected for allexp in kernel method"
	
		b = param.bandparam
		t = 0.25 * b
		r = param.radiusparam
		return b, t, r


class AllInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "inv", label)
	
		#print "parameters selected for allinv in kernel method"
	
		b = param.bandparam
		t = 0.25 * b
		r = param.radiusparam
		return b, t, r


class AllLin:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
	
		#print "parameters selected for alllin in kernel method"
	
		b = param.bandparam
		t = 0.25 * b
		r = param.radiusparam
		return b, t, r


class ExpInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		
		row, col = U.shape
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "exp", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "inv", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r


class ExpLin:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "exp", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "lin", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r


class InvExp:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "inv", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "exp", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r


class InvLin:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "inv", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "lin", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r


class LinExp:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "exp", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r


class LinInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "inv", label)
		
		b = param.bandparam
		t = 0.25 * b
		r = param1.radiusparam
		return b, t, r

class LinConstInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "inv", label)
		
		b = param.bandparam
		t = 0.01
		r = param1.radiusparam
		return b, t, r

class LinIncInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "inv", label)
		
		b = param.bandparam
		t = 10 * b
		r = param1.radiusparam
		return b, t, r

class LinDecInv:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		param1 = BandSelection(self.d, self.num_iters)
		param1.param_calc(dotdistance, k, "inv", label)
		
		b = param.bandparam
		t = 0.01 * b
		r = param1.radiusparam
		return b, t, r

class LinConst:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		#param1 = BandSelection(self.d, self.num_iters)
		#param1.param_calc(dotdistance, k, "inv")
		
		b = param.bandparam
		t = 0.25 * b
		r = sqrt(2)
		return b, t, r

class LinMin:
	
	def set_parameters(self, U, k, label):
		# Compute distances and standard deviation
		row, col = U.shape
		
		dotdistance = zeros((row,))
		for i in range(row):
			dotdistance[i] = abs(self.margin(U[i]))
		
		# Parameters for iteration k
		param = BandSelection(self.d, self.num_iters)
		param.param_calc(dotdistance, k, "lin", label)
		#param1 = BandSelection(self.d, self.num_iters)
		#param1.param_calc(dotdistance, k, "inv")
		
		b = param.bandparam
		t = 0.25 * b
		r = dotdistance[0]
		return b, t, r

class LinearNoiseMethodAllExpConstantsBatch(MarginBasedBasic, AllExp):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodAllExpConstantsBatchOR(MarginBasedOutlierRemoval, AllExp):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodAllInvConstantsBatch(MarginBasedBasic, AllInv):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodAllInvConstantsBatchOR(MarginBasedOutlierRemoval, AllInv):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodAllLinConstantsBatch(MarginBasedBasic, AllLin):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodAllLinConstantsBatchOR(MarginBasedOutlierRemoval, AllLin):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodExpInvConstantsBatch(MarginBasedBasic, ExpInv):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodExpInvConstantsBatchOR(MarginBasedOutlierRemoval, ExpInv):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodExpLinConstantsBatch(MarginBasedBasic, ExpLin):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodExpLinConstantsBatchOR(MarginBasedOutlierRemoval, ExpLin):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodInvExpConstantsBatch(MarginBasedBasic, InvExp):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodInvExpConstantsBatchOR(MarginBasedOutlierRemoval, InvExp):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodInvLinConstantsBatch(MarginBasedBasic, InvLin):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodInvLinConstantsBatchOR(MarginBasedOutlierRemoval, InvLin):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodLinExpConstantsBatch(MarginBasedBasic, LinExp):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodLinExpConstantsBatchOR(MarginBasedOutlierRemoval, LinExp):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class LinearNoiseMethodLinInvConstantsBatch(MarginBasedBasic, LinInv):
	def __init__(self, d, num_iters):
		MarginBasedBasic.__init__(self, d, num_iters)


class LinearNoiseMethodLinInvConstantsBatchOR(MarginBasedOutlierRemoval, LinInv):
	def __init__(self, d, num_iters):
		MarginBasedOutlierRemoval.__init__(self, d, num_iters)


class OptimalSeparator(LinearLearner, ActiveBatchLearner):
	def __init__(self, d, mean1, mean2):
		self.center1 = mean1
		self.center2 = mean2
		self.d = d
    
	def active_batch_train(self, U, ids, oracle, label_budget):

		self.w = self.center1 - self.center2
		self.w /= norm(self.w,2)


################################################################################
# The following is the kernel-based version that could probably replace much of
# the above code once we confirm that it works well.

class KernelMarginBasedActiveLearnerBase(KernelLearner, ActiveBatchLearner):
    
	def __init__(self, d, kernel, num_iters):
		KernelLearner.__init__(self, d, kernel)
		self.num_iters = num_iters
		self.separators = []
		self.pointsusage = 0
		self.combine_final = False
	
	def initialize_weights(self, X, Y):
		# Use averaging algo
		self.support = [[Y[i], X[i]] for i in range(len(Y))]
	
	def hinge_loss_minimization(self, X, Y, tau, radius, prevw):
		
		if SOLVER == 'cvxopt':
			qp = KernelQPwithLinearBand(self.d, self.kernel)
			qp.train(X, Y, tau, radius, prevw)
			self.support = qp.support
		
		elif SOLVER == 'sgd':
			sgd = KernelHingeLossSGD2(self.d, tau)
			sgd.batch_train(X, Y)
			self.support = sgd.support
		
		else:
			raise ValueError, 'Solver %s not implemented.' % SOLVER
	
	def active_batch_train(self, U, ids, oracle, label_budget):
		
		itr = self.num_iters
		count = len(U)
		
		# Set up number of examples in each iteration
		m, n = self.compute_sizes(count, label_budget, itr)
		'''
		# Print some initialization info
		print
		print self.__class__.__name__
		print 'Iterations:', itr
		
		print 'm:', m
		print 'n:', n
		'''
		# access to the training examples and their labels		
		X = array([]).reshape((0, self.d))
		Y = array([]).reshape((0,1))
		used = set()
		
		for i in range(m[0]):
			X = vstack((X, U[i]))
			Y = vstack((Y, array([oracle(ids[i])]).reshape(1,1)))
			used.add(ids[i])
		
		# Pick starting weight vector based on initial sample
		self.initialize_weights(X, Y)
		
		# Keep track of all labeled data
		R = array([]).reshape((0, self.d))
		S = array([]).reshape((0, 1))
		
		for k in range(1, itr):
		
			b, t, r = self.set_parameters(U, k)
			
			if b == 0:
				t = pow(0.1,k)
			
			# Print current iteration info
			#print
			#print 'k:', k
			#print 'b:', b
			#print 't:', t
			#print 'r:', r
			
			# Set of points to send to outlier removal
			P = array([]).reshape((0,self.d))
			workingset = []
			
			# Find n[k] additional points within the band
			while (len(workingset) < n[k]) and (len(used) < count):
				next = len(used)
				used.add(ids[next])
				if abs(self.margin(U[next])) <= b:
					P = vstack((P, U[next]))
					workingset.append(ids[next])
			
			# Check unlabeled data usage and break if necessary
			if len(workingset) != n[k]:
				print
				print 'Out of unlabled data.'
				break
			
			# Perform outlier removal on P
			chosen = self.outlier_removal(P, m[k], b, t)
			assert len(chosen) == m[k]
			
			# Query labels of the selected points
			X = P[chosen]
			Y = array([]).reshape((0, 1))
			for i in chosen:
				Y = vstack((Y, array([oracle(workingset[i])]).reshape(1,1)))
			
			R = vstack((R, X))
			S = vstack((S, Y))
			
			# Perform constrained hinge loss minimization on X, Y
			self.hinge_loss_minimization(X, Y, t, r, self.support)
			
			# Print some end of iteration info?
		
		# Calculate the final weight vector based on points labeled so far.
		if self.combine_final:
			for i in range(len(workingset)):
				if len(S) + m[0] < label_budget:
					R = vstack((R, P[i]))
					S = vstack((S, array([oracle(workingset[i])]).reshape(1,1)))
				else:
					break
		
			self.hinge_loss_minimization(R, S, t, r, self.support)
		
		self.pointsusage = len(S) + m[0]
		'''
		# Print final info
		print
		print 'Label usage:'
		print 'budget:', label_budget
		print 'used:  ', self.pointsusage
		print
		print 'Unlabeled usage:'
		print 'total:', count
		print 'used: ', len(used)
        '''

class KernelMarginBasedBasic(KernelMarginBasedActiveLearnerBase):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedActiveLearnerBase.__init__(self, d, kernel, num_iters)
	
	def compute_sizes(self, unlabeled, labeled, num_iters):
		m = [0] * num_iters
		n = [0] * num_iters
		
		# Increase m_i logarithmically
		denom = sum(log(i + 2) for i in range(num_iters))
		for i in range(num_iters):
			m[i] = int(labeled * log(i + 2) / denom)

		# Add remaining label budget to last iteration
		m[num_iters - 1] += labeled - sum(m)
		
		# Default is no outlier removal, so n = m
		for i in range(num_iters):
			n[i] = m[i]
		
		return m, n

	def outlier_removal(self, P, m, b, tau):
		# No outlier removal. Return all indices.
		return array(range(m))


class KernelMarginBasedOutlierRemoval(KernelMarginBasedActiveLearnerBase):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedActiveLearnerBase.__init__(self, d, kernel, num_iters)
	
	def compute_sizes(self, unlabeled, labeled, num_iters):
		m = [0] * num_iters
		n = [0] * num_iters
		
		# Increase m_i logarithmically
		denom = sum(log(i + 2) for i in range(num_iters))
		for i in range(num_iters):
			m[i] = int(labeled * log(i + 2) / denom)

		# Add remaining label budget to last iteration
		m[num_iters - 1] += labeled - sum(m)
		
		# Number of unlabeled examples available for outlier removal
		#n_total = 0.1 * unlabeled
		n_total = 10 * labeled
		
		# Increase n_i exponentially
		denom = sum(2.0**i for i in range(0, num_iters))
		for i in range(num_iters):
			n[i] = int(n_total * 2.0**i / denom) + m[i]
		
		return m, n

	def outlier_removal(self, P, m, b, tau):
	    # TODO: Fix this to remove dependence on w and r.
		n = P.shape[0]

		# Compute distribution Q over examples in P
		outlier = OutlierRemoval(self.d)
		outlier.train(P, b, r, tau, w, 0.1)
		Q = outlier.weightdist
		Q /= lin.norm(Q, 1)
		
		# Sample random indices based on resulting Q
		chosen = nprandom.choice(n, size = m, replace = False, p = Q)
		return chosen


class KernelMarginBasedTheoreticalParams(KernelMarginBasedBasic, Theoretical):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelMarginBasedTheoreticalParamsOR(KernelMarginBasedOutlierRemoval, Theoretical):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodVarianceBatch(KernelMarginBasedBasic, VarianceMethod):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodVarianceBatchOR(KernelMarginBasedOutlierRemoval, VarianceMethod):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodAllExpConstantsBatch(KernelMarginBasedBasic, AllExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class KernelLinearNoiseMethodAllExpConstantsBatchOR(KernelMarginBasedOutlierRemoval, AllExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodAllInvConstantsBatch(KernelMarginBasedBasic, AllInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodAllInvConstantsBatchOR(KernelMarginBasedOutlierRemoval, AllInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodAllLinConstantsBatch(KernelMarginBasedBasic, AllLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodAllLinConstantsBatchOR(KernelMarginBasedOutlierRemoval, AllLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodExpInvConstantsBatch(KernelMarginBasedBasic, ExpInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodExpInvConstantsBatchOR(KernelMarginBasedOutlierRemoval, ExpInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodExpLinConstantsBatch(KernelMarginBasedBasic, ExpLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodExpLinConstantsBatchOR(KernelMarginBasedOutlierRemoval, ExpLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodInvExpConstantsBatch(KernelMarginBasedBasic, InvExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodInvExpConstantsBatchOR(KernelMarginBasedOutlierRemoval, InvExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodInvLinConstantsBatch(KernelMarginBasedBasic, InvLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodInvLinConstantsBatchOR(KernelMarginBasedOutlierRemoval, InvLin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodLinExpConstantsBatch(KernelMarginBasedBasic, LinExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodLinExpConstantsBatchOR(KernelMarginBasedOutlierRemoval, LinExp):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodLinInvConstantsBatch(KernelMarginBasedBasic, LinInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)


class KernelLinearNoiseMethodLinInvConstantsBatchOR(KernelMarginBasedOutlierRemoval, LinInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedOutlierRemoval.__init__(self, d, kernel, num_iters)

# Added classes for testing

class KernelLinearNoiseMethodLinConstInvConstantsBatch(KernelMarginBasedBasic, LinConstInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class KernelLinearNoiseMethodLinIncInvConstantsBatch(KernelMarginBasedBasic, LinIncInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class KernelLinearNoiseMethodLinDecInvConstantsBatch(KernelMarginBasedBasic, LinDecInv):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class KernelLinearNoiseMethodLinConstConstantsBatch(KernelMarginBasedBasic, LinConst):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class KernelLinearNoiseMethodLinMinConstantsBatch(KernelMarginBasedBasic, LinMin):
	def __init__(self, d, kernel, num_iters):
		KernelMarginBasedBasic.__init__(self, d, kernel, num_iters)

class ActiveKernelQP(KernelLearner, ActiveBatchLearner):
	
	def __init__(self, d, C, kernel):
		KernelLearner.__init__(self, d, kernel)
		self.C = C
	
	def active_batch_train(self, U, ids, oracle, label_budget):
		
		m = min(len(U), label_budget)
		
		X = zeros((m,self.d))
		Y = zeros((m,1))
		
		for i in range(m):
			X[i] = U[i]
			Y[i] = oracle(ids[i])
		
		qp = KernelQP(self.d, self.kernel)
		qp.train(X, Y, self.C)
		self.support = qp.support

class ActiveKernelQPwithLinearBand(KernelLearner, ActiveBatchLearner):
	
	def __init__(self, d, C, kernel):
		KernelLearner.__init__(self, d, kernel)
		self.C = C
	
	def active_batch_train(self, U, ids, oracle, label_budget):
		
		m = min(len(U), label_budget)
		
		X = zeros((m,self.d))
		Y = zeros((m,1))
		
		for i in range(m):
			X[i] = U[i]
			Y[i] = oracle(ids[i])
		
		# Randomly initialize support values
		self.support = [[0.5*Y[i], X[i]] for i in range(len(Y))]
  
		#b, t, r = self.set_parameters(U, 1)
		r = sqrt(2);
		
		qp = KernelQPwithLinearBand(self.d, self.kernel)
		qp.train(X, Y, self.C, r, self.support)
		self.support = qp.support

class KernelPassiveSVM(KernelLearner, ActiveBatchLearner, ActiveSourceLearner):
    
    def __init__(self, d, C, kernel):
        KernelLearner.__init__(self, d, kernel)
        self.C = C
     
    def active_batch_train(self, U, ids, oracle, label_budget):
        m = min(len(U), label_budget)
        
        # Query the labels of m examples
        Y = array([oracle(ids[i]) for i in range(m)]).reshape((m, 1))
        
        # Run standard SVM on the labeled data
        svm = Kernel_soft_SVM(self.d, self.C, self.kernel)
        svm.batch_train(U[:m], Y)
        
        # Use the separator found by SVM
        self.support = svm.support
    
    def active_source_train(self, source, oracle, label_budget):
        U, ids = source(label_budget)
        self.active_batch_train(U, ids, oracle, label_budget)

class KernelSimpleMarginSoftSVMBatch(KernelLearner, ActiveBatchLearner):
    
    def __init__(self, d, C, kernel):
        KernelLearner.__init__(self, d, kernel)
        self.C = C
        self.initial_sample = 5
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        
        # Create holders for the labeled data
        X = array([]).reshape((0, self.d))
        Y = array([]).reshape((0, 1))
        used = set()
        labeled = []
        
        # Query the labels of some examples
        start = min(self.initial_sample, label_budget)
        for i in range(start):
            X = vstack((X, U[i]))
            Y = vstack((Y, array([oracle(ids[i])]).reshape((1,1))))
            used.add(ids[i])
            labeled.append(ids[i])
        
        # Run standard SVM on the labeled data
        #svm = StochasticDualCoordinateAscent(self.d, self.C, self.kernel)
        #svm.train(X, Y)
        svm = Kernel_soft_SVM(self.d, self.C, self.kernel)
        svm.batch_train(X, Y)
        
        # Until the label budget is reached
        while len(used) < label_budget:
            
            margins = []
            
            # Find the unlabeled example with the smallest margin
            for i in range(len(U)):
                margins.append(abs(svm.margin(U[i])))
            
            min_margin = inf
            min_index = 0
            for i in range(len(U)):
                if ids[i] in used:
                    continue
                cur_margin = margins[i]
                if cur_margin < min_margin:
                    min_margin = cur_margin
                    min_index = i
            
            # Query its label
            X = vstack((X, U[min_index]))
            Y = vstack((Y, array([oracle(ids[min_index])]).reshape((1,1))))
            used.add(ids[min_index])
			
			# Run SVM on all the labeled data
            svm.batch_train(X, Y)
        
        # Use the most recent separator found by SVM
        self.support = svm.support

class PassiveSVMDual(KernelLearner, ActiveBatchLearner):
    
    def __init__(self, d, C, kernel):
        KernelLearner.__init__(self, d, kernel)
        self.C = C
     
    def active_batch_train(self, U, ids, oracle, label_budget):
        m = min(len(U), label_budget)
        
        # Query the labels of m examples
        Y = array([oracle(ids[i]) for i in range(m)]).reshape((m, 1))
        
        # Run standard SVM on the labeled data
        svm = StochasticDualCoordinateAscent(self.d, self.C, self.kernel)
        svm.train(U[:m], Y)
        
        # Use the separator found by SVM
        self.support = svm.support


class SVMSGD(LinearLearner, ActiveBatchLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d)
        self.C = C
     
    def active_batch_train(self, U, ids, oracle, label_budget):
        m = min(len(U), label_budget)
        
        # Query the labels of m examples
        Y = array([oracle(ids[i]) for i in range(m)]).reshape((m, 1))
        
        # Run standard SVM on the labeled data
        svm = HingeLossSGD(self.d, self.C)
        svm.batch_train(U[:m], Y)
        
        # Use the separator found by SVM
        self.w = svm.w
