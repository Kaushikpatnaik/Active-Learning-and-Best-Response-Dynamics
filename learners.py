from numpy import *



class Learner(object):
    
    def __str__(self):
        return self.__class__.__name__


class LinearLearner(Learner):
    
    def __init__(self, d, w = None):
        self.d = d
        self.w = w
        if self.w is None:
            self.w = zeros(self.d)
    
    def classify(self, x):
        return sign(dot(self.w, x))
    
    def margin(self, x):
        return dot(x, self.w)


class KernelLearner(Learner):
    
    def __init__(self, d, kernel):
        self.d = d
        self.kernel = kernel
        self.support = []  # List of (coeff, point) pairs
    
    def classify(self, x):
        return sign(self.margin(x))
    
    def margin(self, x):
        return sum(coeff * self.kernel(point, x) for coeff, point in self.support)


class OnlineLearner(Learner):
    
    def update(self, x, y):
        raise NotImplementedError


class PassiveSupervisedLearner(Learner):
    
    def batch_train(self, X, Y):
        raise NotImplementedError


class SemiSupervisedLearner(Learner):
    
    def ssl_train(self, U, L, Y):
        raise NotImplementedError


class ActiveBatchLearner(Learner):
    
    def active_batch_train(self, U, ids, oracle, label_budget):
        raise NotImplementedError


class ActiveSourceLearner(Learner):
    
    def active_source_train(self, source, oracle, label_budget):
        raise NotImplementedError










